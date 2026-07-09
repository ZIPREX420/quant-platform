"""validation module: CSV auto-detection, metrics math, Monte Carlo determinism."""
import pytest

from quant_platform.validation import Trade, TradeListError, load_trades_csv, monte_carlo, trade_metrics
from quant_platform.validation.analysis import sharpe_like


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestCsvLoading:
    def test_return_fraction_column(self, tmp_path):
        p = write(tmp_path, "t.csv", "return_pct\n0.02\n-0.01\n")
        trades = load_trades_csv(p)
        assert [t.return_fraction for t in trades] == [0.02, -0.01]

    def test_percent_scale_detected_per_file(self, tmp_path):
        p = write(tmp_path, "t.csv", "return_pct\n2.5\n-1.0\n")
        trades = load_trades_csv(p)
        assert [t.return_fraction for t in trades] == [0.025, -0.01]

    def test_pnl_plus_notional(self, tmp_path):
        p = write(tmp_path, "t.csv", "pnl,notional\n20,1000\n-10,1000\n")
        trades = load_trades_csv(p)
        assert [t.return_fraction for t in trades] == [0.02, -0.01]

    def test_ambiguous_columns_rejected(self, tmp_path):
        p = write(tmp_path, "t.csv", "return_pct,profit_pct\n1,2\n")
        with pytest.raises(TradeListError, match="ambiguous"):
            load_trades_csv(p)

    def test_unusable_columns_explained(self, tmp_path):
        p = write(tmp_path, "t.csv", "foo,bar\n1,2\n")
        with pytest.raises(TradeListError, match="no usable columns"):
            load_trades_csv(p)

    def test_empty_and_bad_rows(self, tmp_path):
        with pytest.raises(TradeListError, match="no trade rows"):
            load_trades_csv(write(tmp_path, "e.csv", "return_pct\n"))
        with pytest.raises(TradeListError, match="bad value"):
            load_trades_csv(write(tmp_path, "b.csv", "return_pct\nxyz\n"))

    def test_header_case_and_bom(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_bytes(b"\xef\xbb\xbfReturn_Pct\n0.01\n")
        assert load_trades_csv(p)[0].return_fraction == 0.01


class TestMetrics:
    def test_known_values(self):
        trades = [Trade(0.10), Trade(-0.05), Trade(0.10), Trade(-0.05)]
        m = trade_metrics(trades)
        assert m.trades == 4 and m.win_rate_pct == 50.0
        assert m.profit_factor == 2.0  # 0.20 gross win / 0.10 gross loss
        assert m.expectancy_pct == 2.5
        # equity: 1.1 * .95 * 1.1 * .95 = 1.091... total ~ +9.2%
        assert m.total_return_pct == pytest.approx(9.2, abs=0.1)
        assert m.max_drawdown_pct == -5.0  # each loss is a 5% dip from peak

    def test_all_winners_pf_none(self):
        m = trade_metrics([Trade(0.01), Trade(0.02)])
        assert m.profit_factor is None and m.max_drawdown_pct == 0.0

    def test_sharpe_like(self):
        assert sharpe_like([Trade(0.01), Trade(0.01)], 12) is None  # zero stdev
        s = sharpe_like([Trade(0.02), Trade(-0.01), Trade(0.03)], 12)
        assert isinstance(s, float)


class TestMonteCarlo:
    TRADES = [Trade(r) for r in (0.05, -0.02, 0.03, -0.04, 0.06, -0.01,
                                 0.02, -0.03, 0.04, -0.02, 0.05, -0.05)]

    def test_deterministic_given_seed(self):
        a = monte_carlo(self.TRADES, runs=200, seed=7)
        b = monte_carlo(self.TRADES, runs=200, seed=7)
        assert a == b

    def test_percentile_ordering(self):
        mc = monte_carlo(self.TRADES, runs=500)
        assert mc.terminal_return_pct_p05 <= mc.terminal_return_pct_p50
        assert mc.max_drawdown_pct_p05 <= mc.max_drawdown_pct_p50 <= 0.0
        assert 0.0 <= mc.prob_negative_terminal_pct <= 100.0

    def test_minimum_trades_enforced(self):
        with pytest.raises(ValueError, match=">= 10 trades"):
            monte_carlo([Trade(0.01)] * 9)


class TestReportRendering:
    def test_sections_contain_criteria(self, tmp_path):
        from quant_platform.validation.cli import render_report_sections
        m = trade_metrics(TestMonteCarlo.TRADES)
        mc = monte_carlo(TestMonteCarlo.TRADES, runs=100)
        text = render_report_sections("t.csv", m, mc, 1.2)
        assert "Monte Carlo (100 bootstrap runs" in text
        assert "declared risk caps" in text and "Profit factor" in text
