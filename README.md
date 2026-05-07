# smc-strategy-algo
BTC SMC Algo Strategy — Smart Money Concepts Backtester

An algorithmic trading strategy for Bitcoin built on Smart Money Concepts (SMC) — the same institutional framework used by professional traders to track "smart money" (banks & hedge funds) movement in markets.
Smart Money Concepts is a price action framework based on the idea that large institutional players (banks, hedge funds) leave footprints in the market through specific patterns. By identifying these patterns — like Order Blocks, Fair Value Gaps, and Liquidity Sweeps — retail traders can align with institutional flow instead of trading against it.
SMC Concepts Implemented
ConceptDescription
Break of Structure (BOS)Confirms trend continuation when price breaks a previous swing pointChange of Character 
(CHoCH)Signals potential trend reversal — key entry triggerOrder Blocks (OB)Last opposing candle before a strong impulse move — institutional supply/demand zonesFair Value Gaps (FVG)3-candle imbalance zones where price tends to retraceLiquidity SweepsDetection of equal highs/lows being swept before reversalMarket StructureFull HH / HL / LH / LL labeling for trend identificationPremium / Discount ZonesFibonacci 50% filter — only long in discount, short in premium.
HOW TO RUN--
1. Install dependencies
bashpip install pandas numpy matplotlib yfinance scipy
2. Run with defaults (BTC-USD, 30m, last 30 days)
bashpython smc_strategy.py
3. Custom run
bashpython smc_strategy.py --symbol BTC-USD --interval 1h --start 2024-01-01 --end 2024-06-01
