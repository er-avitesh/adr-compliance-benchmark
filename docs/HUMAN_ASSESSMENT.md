# Human Assessment Evidence

The primary reference file is `results/human_ground_truth.json`. It contains
assessments from seven software and IT industry experts with at least eight
years of professional experience and practical exposure to ADRs or architecture
decision documentation.

Generated public releases replace reviewer names with stable codes (`P01`-`P07`
for the primary assessment and `S01`-`S03` for the second assessment). The
codes preserve assignment patterns without publishing personal identifiers.
The public primary file is limited to the 162 evaluation ADRs and four declared
holdouts; no scores or notes for those records are changed.

`results/human_ground_truth_interrator.json` contains a separate second human
assessment for 41 ADRs in the fixed evaluation set. The spelling of the file
name is retained to preserve existing hashes and references.

## Reproducible Counts

| Comparison | Matches | Total | Agreement |
| --- | ---: | ---: | ---: |
| Recorded overall label, primary versus second | 37 | 41 | 90.2% |
| Criterion-derived overall class, primary versus second | 27 | 41 | 65.9% |
| Criterion-derived SC class | 28 | 41 | 68.3% |
| Criterion-derived DQ class | 24 | 41 | 58.5% |

Eleven recorded overall labels in the second file differ from the class
obtained by applying the frozen formula to the C1-C7 and Q1-Q7 values in the
same record. Primary recorded overall labels are formula-consistent for these
41 records.

The analysis reports the recorded-label and criterion-derived results
separately. It does not replace submitted overall labels, infer missing
procedural records, or treat 90.2% recorded-label agreement as criterion-level
agreement.

## Scope of the Retained Evidence

The JSON files retain reviewer identifiers, stated experience, criteria,
scores, overall labels, and notes. They do not independently establish the
assignment mechanism, blinding procedure, review timestamps, or disagreement
adjudication. Claims about those procedures require separate source records.
