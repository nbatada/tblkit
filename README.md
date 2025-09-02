# tblkit

Small, fast CLI for everyday table wrangling. Works with CSV/TSV (and friends), streams nicely, and plays well with Unix pipes.


tblkit fills the gap between Unix text tools and “fire up pandas”: ergonomic defaults, safe cleaning, and the few “you always need them” behaviors (non-folding view, numeric/date sorting, fuzzy join).


---

## Install

```bash
# recommended
pip install tblkit

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

<!-- START: TBLKIT COMMANDS -->
<details>
<summary><code>tblkit --commands</code> (click to expand)</summary>

```text
tblkit
├── col                         (Column operations)
│   ├── add                     (Add a new column)
│   ├── clean                   (Normalize string values in selected columns.)
│   ├── drop                    (Drop columns by name/glob/position/regex)
│   ├── extract                 (Extract regex groups into new columns.)
│   ├── join                    (Join values from multiple columns into a new column.)
│   ├── move                    (Reorder columns by moving a selection.)
│   ├── rename                  (Rename column(s) via map string)
│   ├── replace                 (Value replacement in selected columns.)
│   ├── split                   (Split a column by pattern into multiple columns)
│   ├── strip                   (Trim/squeeze whitespace; optional substring/fixed-count strip.)
│   └── subset                  (Select a subset of columns by name/glob/position/regex)
├── header                      (Header operations)
│   ├── add                     (Add a generated header to a headerless file.)
│   ├── add-prefix              (Add a fixed prefix to columns.)
│   ├── add-suffix              (Add a fixed suffix to columns.)
│   ├── clean                   (Normalize all column names (deprecated; use: tbl clean))
│   ├── prefix-num              (Prefix headers with 1_, 2_, ... (or custom fmt).)
│   ├── rename                  (Rename headers via map string or file)
│   └── view                    (View header column names)
├── row                         (Row operations)
│   ├── add                     (Add a row with specified values.)
│   ├── drop                    (Drop rows by 1-based index.)
│   ├── grep                    (Filter rows by a list of words or phrases.)
│   ├── head                    (Select first N rows)
│   ├── sample                  (Randomly sample rows)
│   ├── shuffle                 (Randomly shuffle all rows.)
│   ├── subset                  (Select a subset of rows using a query expression)
│   ├── tail                    (Select last N rows)
│   └── unique                  (Filter unique or duplicate rows)
├── sort                        (Sort rows or columns)
│   ├── cols                    (Sort columns by their names)
│   └── rows                    (Sort rows by column values)
├── tbl                         (Whole-table operations)
│   ├── aggregate               (Group and aggregate data)
│   ├── clean                   (Clean headers and string values throughout the table.)
│   ├── concat                  (Concatenate piped table with other files)
│   ├── frequency               (Show top N values per column.)
│   ├── join                    (Relational join between two tables.)
│   ├── melt                    (Melt table to long format.)
│   ├── pivot                   (Pivot a table from long to wide format)
│   ├── sort                    (Sort rows by column values (alias for 'sort rows').)
│   ├── squash                  (Group rows and squash column values into delimited strings.)
│   └── transpose               (Transpose the table.)
└── view                        (Pretty-print a table (ASCII, non-folding).)

```
</details>
<!-- END: TBLKIT COMMANDS -->



---

## Command map

```bash
tblkit --commands      # show the command tree
tblkit --help          # global help
tblkit view --help     # pretty printer options
tblkit tbl --help      # whole-table operations (clean, join, sort, …)
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