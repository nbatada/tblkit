# tblkit Usage Cookbook (Extensive) — 2025-09-03

This document shows **non‑trivial, realistic examples** for **every current command** in `tblkit`
(based on the attached `core.py`). It also includes a candid **brainstorm and roadmap** for **tblkit 2.0**.

> Conventions
>
> - Input files are examples – replace with your paths.  
> - `--sep csv` means CSV input; default is TAB.  
> - Column selectors accept names, globs (`val*`), ranges (`2-5`), and regex (`re:tmp_`).  
> - All commands read from stdin or `-i/--input` and write to stdout by default.

---

## Command Index (from current code)

- **view** — Pretty-print a table (ASCII, non-folding).
- **tbl** — Whole-table operations:
  - `clean`, `frequency`, `join`, `sort` (alias of `sort rows`), `pivot`, `concat`, `aggregate`, `collapse`, `melt`, `transpose`
- **col** — Column operations:
  - `subset`, `clean`, `drop`, `rename`, `replace`, `strip`, `move`, `extract`, `split`, `add`, `join`
- **header** — Header operations:
  - `view`, `rename`, `add`, `clean` (deprecated), `prefix-num`, `add-prefix`, `add-suffix`
- **row** — Row operations:
  - `subset`, `grep`, `head`, `tail`, `sample`, `unique`, `shuffle`, `drop`, `add`
- **sort** — Sorting:
  - `rows`, `cols`

---

# 1) `view` — Pretty-print

### 1.1 Pretty-print a wide table with truncation control
```bash
cat results.tsv \
| tblkit view --max-col-width 60 --max-cols 20
```
**Why:** Readable terminal output; useful as the final step of a pipeline.

### 1.2 Focus on specific columns (rich selection)
```bash
cat results.tsv \
| tblkit view -c 'id,run*,re:^metric_,re:date$' --show-full
```
**Why:** Curate what you actually need to see; `--show-full` disables truncation when auditing a few fields.

---

# 2) `tbl` — Whole-table operations

## 2.1 `tbl clean` — sanitize headers & values
**Task:** Normalize headers, dedupe, and standardize string values while leaving some columns untouched.
```bash
cat cohort.tsv \
| tblkit tbl clean \
    --case lower --spaces _ --ascii \
    --dedupe _ --exclude notes,free_text \
| tblkit view -c 'sample_id,cohort,batch,platform,mean_coverage'
```
**Why:** Reproducible “make it tidy” step: stable header names, consistent casing, and value normalization.

## 2.2 `tbl frequency` — value frequencies (top-N)
**Task:** Audit categorical drift across releases.
```bash
cat releases.tsv \
| tblkit tbl frequency -c 'status,platform,center' -n 10 --all-columns \
| tblkit view --max-col-width 40
```
**Why:** Quick distribution checks for data validation.

## 2.3 `tbl join` — exact + **fuzzy** join with key normalization
**Task:** Join a cohort sheet to QC metrics where IDs are messy (`PT-0007` vs `pt7`).
```bash
tblkit tbl join \
  --left cohort.csv --right qc.tsv --sep csv \
  --keys sample_id --how left \
  --keep-left sample_id,cohort,batch,platform \
  --keep-right mean_coverage,dup_rate,contam_est \
  --key-norm lower --key-norm strip_prefix:PT- --key-norm strip:_- --key-norm rm_leading_zeros --key-norm trim \
  --fuzzy --fuzzy-threshold 0.92 --require-coverage \
  --report join_report.csv \
| tblkit view --max-col-width 60
```
**Why:** Built-in normalizers + fuzzy fallback cover real-world ID chaos; `--report` documents matches.

## 2.4 `tbl sort` — alias to `sort rows`
```bash
cat metrics.csv \
| tblkit tbl sort --by service,day --date --natural \
| tblkit view
```
**Why:** Keep the whole-table mental model; same options as `sort rows`.

## 2.5 `tbl pivot` — wide pivot
**Task:** Gene × impact counts to wide matrix (after pre-aggregation).
```bash
cat counts.tsv \
| tblkit tbl pivot --index gene --columns impact --values n \
| tblkit view
```
**Why:** Produce matrices for analysts/review.

## 2.6 `tbl concat` — vertical union with schema drift
```bash
tblkit tbl concat --filelist runs.txt --fill-missing \
| tblkit view -c 'sample_id,dup_rate,mean_coverage,insert_mean'
```
> `runs.txt` contains one path per line. `--fill-missing` unions columns across files.

## 2.7 `tbl aggregate` — flexible groupby
**Variant A: multiple functions per column**
```bash
cat events.csv \
| tblkit tbl aggregate --group-by user_id \
    --columns latency_ms --funcs mean,max,p95 \
| tblkit view
```
**Variant B: per-column ops**
```bash
cat events.csv \
| tblkit tbl aggregate --group-by service \
    --ops 'latency_ms:p95,errors:sum,rps:mean' \
| tblkit view
```
**Why:** Both styles are supported: `--funcs` or `--ops`.

## 2.8 `tbl collapse` — group, collect, and join values
**Task:** For each user, collect distinct slow endpoints.
```bash
cat events.csv \
| tblkit row subset 'latency_ms > 250' --sep , \
| tblkit tbl collapse -g user_id -d ';' \
| tblkit view -c 'user_id,collapsed'
```
**Why:** Produce compact “evidence” columns downstream can read.

## 2.9 `tbl melt` — wide → long
```bash
cat qc_panel.tsv \
| tblkit tbl melt --id_vars sample_id --value_vars dup_rate,mean_coverage,insert_mean \
    --var_name metric --value_name value \
| tblkit view --max-col-width 40
```

## 2.10 `tbl transpose`
```bash
cat keyvals.tsv | tblkit tbl transpose | tblkit view
```

---

# 3) `col` — Column operations

## 3.1 `col subset` — powerful selectors
```bash
cat big.tsv \
| tblkit col subset -c 'id,run*,2-5,re:^aux_' \
| tblkit view
```

## 3.2 `col clean` — normalize string values
```bash
cat names.tsv \
| tblkit col clean -c 'first_name,last_name' --unicode-nfkc --lower --spaces '_' --ascii \
| tblkit view
```

## 3.3 `col drop` — drop (or invert keep)
```bash
cat telemetry.tsv \
| tblkit col drop -c 're:^debug_,aux_*' --invert --keep-columns 'id,timestamp' \
| tblkit view
```

## 3.4 `col rename` — mapping string
```bash
cat messy.tsv \
| tblkit col rename --map 'old_id:sample_id,My Col:my_col' \
| tblkit view
```

## 3.5 `col replace` — value mapping (with regex option)
```bash
cat phenotypes.tsv \
| tblkit col replace -c status --from 'POS,NEG,UNK' --to '1,0,NA' \
| tblkit view
```
**Regex example**
```bash
cat notes.tsv \
| tblkit col replace -c note_text --from 're:\\s+' --to '_' --regex \
| tblkit view
```

## 3.6 `col strip` — whitespace & substring strip
```bash
cat strings.tsv \
| tblkit col strip -c comments --lstrip-substr '[DRAFT]' --rstrip-substr '(END)' \
| tblkit view
```

## 3.7 `col move` — reorder by anchor
```bash
cat frame.tsv \
| tblkit col move -c 'notes' --before mean_coverage \
| tblkit view
```

## 3.8 `col extract` — named regex groups → new columns
```bash
cat variants.tsv \
| tblkit col extract -c INFO --regex 'gene=(?P<gene>[^;]+);.*impact=(?P<impact>[^;]+)' --drop-source \
| tblkit view
```

## 3.9 `col split` — regex or fixed substring
```bash
cat cohort.tsv \
| tblkit col split -c sample_key --pattern '[:|]' --maxsplit 2 -n source,center,id --inplace \
| tblkit view
```

## 3.10 `col add` — add a new column near another
```bash
cat metrics.tsv \
| tblkit col add -c 'service' --new-header 'env' -v 'prod' \
| tblkit view
```

## 3.11 `col join` — join multiple cols into one
```bash
cat coords.tsv \
| tblkit col join -c 'chrom,pos,ref,alt' -d ':' -o 'variant' --keep \
| tblkit view
```

---

# 4) `header` — Header operations

## 4.1 `header view`
```bash
cat table.tsv | tblkit header view
```

## 4.2 `header rename` — file or inline map
```bash
cat table.tsv \
| tblkit header rename --from-file mapping.tsv \
| tblkit view
```

## 4.3 `header add` — generate when missing (idempotent unless --force)
```bash
cat noheader.tsv \
| tblkit header add --prefix c_ --start 1 \
| tblkit view
```

## 4.4 `header clean` (deprecated; prefer `tbl clean`)
```bash
cat table.tsv | tblkit header clean --case lower --spaces _
```

## 4.5 `header prefix-num`
```bash
cat table.tsv | tblkit header prefix-num --format '{i}_' --start 1
```

## 4.6 `header add-prefix` / `add-suffix`
```bash
cat table.tsv | tblkit header add-prefix --text 'x_' 
cat table.tsv | tblkit header add-suffix --text '_mm'
```

---

# 5) `row` — Row operations

## 5.1 `row subset` — expression filter (Pandas query)
```bash
cat events.csv \
| tblkit row subset 'latency_ms > 250 and env == "prod" and error != "NA"' --sep , \
| tblkit view
```

## 5.2 `row grep` — word/phrase filter (regex optional)
```bash
cat notes.tsv \
| tblkit row grep --word-file phrases.txt -c 'title,body' --ignore-case \
| tblkit view
```

## 5.3 `row head` / `row tail`
```bash
cat table.tsv | tblkit row head -n 50
cat table.tsv | tblkit row tail -n 50
```

## 5.4 `row sample` — deterministic with seed (via global `--seed`)
```bash
tblkit -i table.tsv --seed 42 row sample -n 500 | tblkit view
```

## 5.5 `row unique` — deduplicate by subset of columns
```bash
cat measurements.tsv \
| tblkit row unique -c 'subject_id,visit,metric' \
| tblkit view
```

## 5.6 `row shuffle`
```bash
cat table.tsv | tblkit row shuffle
```

## 5.7 `row drop` — by 1‑based indices / ranges
```bash
cat table.tsv | tblkit row drop --indices '1,3,10-12'
```

## 5.8 `row add` — append or insert
```bash
cat table.tsv \
| tblkit row add --values 'a,b,c,d' --at 1 \
| tblkit view
```

---

# 6) `sort`

## 6.1 `sort rows` — natural/date/numeric
```bash
cat daily.csv \
| tblkit sort rows --by service,day --natural --date \
| tblkit view
```

## 6.2 `sort cols` — header sort
```bash
cat table.tsv | tblkit sort cols --natural | tblkit view
```

---

## Appendix: I/O, separators, and encodings
- `--sep csv` for CSV; otherwise default is TAB.  
- `--encoding` for encodings (default utf-8).  
- NA tokens: `--na-values` (adds to the default list).  
- Pretty output: `--pretty` for aligned printing to terminal.

---

# Weaknesses & Deficiencies (today)

1) **Discoverability & cognitive load**
   - Many commands; flags vary across groups.
   - Help is long; users can’t quickly find “the right flag for this task.”

2) **Schema & types**
   - Weak type inference and surfacing (string vs int vs date).
   - No first-class schema description or validation.

3) **Performance & scale**
   - In‑memory Pandas everywhere; large files cause RAM spikes.
   - Expensive operations (e.g., fuzzy join) are O(N×M) and unbounded.

4) **Error ergonomics**
   - Messages are improving, but still terse. Root‑cause hints are limited.
   - Little “explain why rows/columns changed” tooling.

5) **Streaming gaps**
   - Some tasks (grep/head/tail) stream naturally; others buffer entire tables.
   - No chunked processing for simple transforms.

6) **Date/time & parsing**
   - Ad‑hoc date parsing per command; no central policy, formats, or tz handling.

7) **Testing & reproducibility**
   - No built‑in “golden test” harness or recorded pipelines.
   - Diff‑friendly outputs (sorting/normalizing) aren’t standardized.

8) **Ecosystem integration**
   - No direct Parquet/Feather/Arrow, SQLite/ DuckDB, or S3/GS connectors.
   - Limited plugin discovery and versioning.

9) **Observability**
   - Logging is plaintext; no structured logs or progress bars for long ops.

10) **Security & safety**
    - `row subset` uses Pandas query – potential foot‑guns with user input.
    - No sandboxing for custom expressions.

---

# tblkit 2.0 — Plan to Earn Adoption

### A. UX & Discoverability
- **`tblkit help <task>`**: task‑oriented help that maps “what I want” → pipeline snippets.  
- **`--explain`** mode: show what each step will do (columns touched, estimated rows affected).  
- **Short help by default** (you already trimmed I/O); **`--verbose-help`** for full detail.  
- **`tblkit doctor`**: sample data, guess delimiter/NA/dtypes, and recommend flags.

### B. Schema & Types
- **Schema engine**: infer + cache dtypes (with YAML sidecar), expose `tblkit schema show/validate`.  
- **Consistent type options**: `--as-int/--as-float/--as-date fmt=...` shared across commands.  
- **Date policies**: one place for tz/format parsing; `--date` just toggles it.

### C. Engines & Scale
- **Backend abstraction**: Pandas default; **Polars** or **DuckDB** opt‑in for scale.  
- **Chunked transforms** where possible (map/filter/replace/grep).  
- **Fuzzy join v2**: use **rapidfuzz**, blocking (prefix/q‑gram), and **top‑k** candidates with caps.

### D. Reproducibility & CI
- **`tblkit record`** to emit a pipeline YAML while you work; **`tblkit run pipeline.yml`** to replay.  
- Golden‑tests: `tblkit test` runs examples on fixtures and diff‑checks outputs.  
- **Version pinning**: pipeline files record `tblkit` version and engine.

### E. Ecosystem & Formats
- **`tblkit io`** group: Parquet/Feather/JSONL; **S3/GS** URIs.  
- **`tblkit sql`**: small wrapper over DuckDB (SELECT / JOIN / PIVOT in SQL).  
- **Plugins**: discovery (`tblkit --plugins` shows versions), cookiecutter template.

### F. Observability & DX
- **Structured logs** (`--log-format json`) and **progress bars** for long ops.  
- **Profiling**: `--profile` prints per‑step timing & memory (top‑line).  
- **`--dry-run`**: print the plan and sampled preview without executing fully.

### G. Safety
- **Expression sandbox** for `row subset` (whitelist functions).  
- **Quota guards** for fuzzy join (max candidates per key), with clear errors.

---

## Milestones

**M1 (DX & docs)**
- Task‑oriented help, short/verbose help, examples overhaul (this doc).
- `tblkit doctor`, structured logs, progress for long ops.

**M2 (Scale & schema)**
- Engine abstraction (add Polars), schema show/validate, chunked streaming transforms.

**M3 (Joins & pipelines)**
- Fuzzy join v2 (rapidfuzz, blocking), pipeline record/run, golden tests + fixtures.

**M4 (Ecosystem)**
- Parquet/Feather/Arrow, DuckDB `tblkit sql`, cloud URIs, plugin discovery/versioning.

---

## Final note
Keep `tblkit` narrow and excellent: **table algebra at the CLI**.  
Where scale or SQL shines, **delegate instead of re‑implementing**—but make it seamless to combine.

