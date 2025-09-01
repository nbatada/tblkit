# tblkit

Small, fast CLI for everyday table wrangling. Works with CSV/TSV (and friends), streams nicely, and plays well with Unix pipes.

- Pretty, non-folding table view for terminals (`less -S` friendly)
- Clean headers/values (without wrecking numbers or dates)
- Sort by text, **numeric-like strings**, or **dates**
- Joins (incl. **fuzzy** left-join with simple normalizers)
- Column select/split/strip/rename
- Quiet on broken pipes (no stack traces on `| head`)

---

## Install

```bash
# recommended
pipx install tblkit

# or
pip install -U tblkit

# or, from source
pip install git+https://github.com/nbatada/tblkit
```

Requires Python 3.9+.

---

## Quick start

Assume a CSV like:

```
ticker,name,sector,market_cap,last_close,dma200,pct_from_dma200
CNC,Centene Corporation,Health Care,"$14,647,224,469",29.04,52.96,-45.17%
IT,Gartner,Information Technology,"$19,024,075,787",251.19,434.06,-42.13%
TTD,Trade Desk (The),Communication Services,"$26,725,624,743",54.66,84.68,-35.45%
```

```bash
# 1) View (non-folding ASCII; pipe to less -S to scroll horizontally)
cat sp500_below_200dma.csv   | tblkit --sep csv view   | less -S

# 2) Clean headers + values (preserves decimals, percents, and dates; removes thousands)
cat sp500_below_200dma.csv   | tblkit --sep csv tbl clean   | head

# 3) Sort numerically on a currency/commas column (data stays text)
cat sp500_below_200dma.csv   | tblkit --sep csv tbl sort --by market_cap --numeric   | head -n 3

# 4) Sort by date
cat trades.csv   | tblkit --sep csv tbl sort --by trade_date --date

# 5) Fuzzy left join (normalize suffixes like -01; match ≥0.92)
tblkit tbl join   --left A.csv --right B.csv --sep csv   --keys id --how left --fuzzy   --key-norm strip_suffix:-\d+$,rm_leading_zeros,upper   --threshold 0.92 --report fuzzy_report.csv
```

**Tips**

- Global flags like `--sep csv` work **before or after** the command (e.g., `tblkit tbl sort --sep csv ...`).
- Prefer `--output-sep tsv` over `tr ',' '\t'` (CSV quoting is respected on output).

---

## Why tblkit (vs awk/sed/cut/join, csvkit, jq, pandas)?

| Task | awk/sed/cut/join | csvkit | jq | pandas | **tblkit** |
|---|---|---|---|---|---|
| Stream-friendly (pipes) | ✅ | ✅ | ⚠️ (JSON) | ❌ | ✅ |
| Pretty table view (non-folding) | ❌ | ❌ | ❌ | ⚠️ | ✅ |
| Clean headers/values safely | ⚠️ | ⚠️ | ❌ | ✅ | ✅ (text only; keeps numbers/dates) |
| Numeric sort on text numbers (e.g., `$1,234`) | ⚠️ | ⚠️ | ❌ | ✅ | ✅ (`--numeric`) |
| Date sort | ❌ | ⚠️ | ❌ | ✅ | ✅ (`--date`) |
| Fuzzy join | ❌ | ❌ | ❌ | ⚠️ | ✅ (simple, built-in) |
| Low ceremony | ⚠️ | ✅ | ❌ | ❌ | ✅ |

tblkit fills the gap between Unix text tools and “fire up pandas”: ergonomic defaults, safe cleaning, and the few “you always need them” behaviors (non-folding view, numeric/date sorting, fuzzy join).

---

## Command map

```bash
tblkit --commands      # show the command tree
tblkit --help          # global help
tblkit view --help     # pretty printer options
tblkit tbl --help      # whole-table operations (clean, join, sort, …)
```

**Highlights**

- `view` — pretty ASCII table (non-folding). `--max-col-width 40` (default), `--show-full`, `--max-cols N`.
- `tbl clean` — header cleaning + (by default) **string values only**:
  - Keeps decimals, `%`, and dates; removes thousands separators on numeric-like strings.
  - Header defaults: lowercase, squeeze spaces, ASCII-only, strip punctuation.
  - Values: keep punctuation by default; enable removal with `--strip-punct-values`.
  - Exclude columns from value cleaning: `--exclude col1,col2`.
- `tbl sort` / `sort rows` — `--by A,B`, `--descending`, `--natural`, **`--numeric`**, **`--date`** (`--date-format` optional).
- `tbl join` — `--left/--right`, `--keys`, `--how` (inner/left/right/outer). Fuzzy: `--fuzzy`, `--key-norm`, `--threshold`, `--report`, `--require-coverage`.

---

## Configuration

- **Colors**: honors `NO_COLOR` env var. Help/errors display color when attached to a TTY; set `NO_COLOR=1` to force plain text.
- **Broken pipes**: `view`/write operations suppress `BrokenPipeError` so `| head` or exiting `less` doesn’t print tracebacks.

---

## Sample data for testing

```bash
curl -L https://raw.githubusercontent.com/nbatada/tblkit/main/tblkit/examples/sp500_below_200dma.sample.csv \
  | tblkit --sep csv view | head -n 10
```

---

## Contributing & plugins

- **Questions / bugs**: open a GitHub Issue with:
  - the command you ran,
  - expected vs actual output,
  - a minimal sample (≤20 lines).
- **Pull requests**: welcome—please add tests for new behaviors (especially around numeric/date handling and fuzzy joins).
- **Plugins**: tblkit can load domain-specific extensions. Run `tblkit --plugins` to list any installed. If you want to publish a plugin (e.g., for bioinformatics or finance), open an Issue and we’ll point you at the plugin hooks and an example skeleton.

---

## License

[MIT](LICENSE)

---

### Appendix: install notes

- Prefer `pipx` for isolated CLI installs.
- Windows users: run in a terminal that supports UTF-8; set `PYTHONUTF8=1` if needed.
