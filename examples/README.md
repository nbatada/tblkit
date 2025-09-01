# Examples

A tiny sample CSV to try tblkit quickly:

```bash
# View (non-folding)
cat examples/sp500_below_200dma.sample.csv | tblkit --sep csv view | less -S

# Clean headers + values (preserves decimals/percents, removes thousands)
cat examples/sp500_below_200dma.sample.csv | tblkit --sep csv tbl clean | head

# Numeric sort on a currency column without changing the data
cat examples/sp500_below_200dma.sample.csv | tblkit --sep csv tbl sort --by market_cap --numeric | head -n 3
```
