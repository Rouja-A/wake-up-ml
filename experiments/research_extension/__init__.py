"""Research extension: streaming-wake-up-network exploratory study.

Isolated from the frozen primary assignment (Models A/B/C, official-test evaluation). Nothing in this
package imports scripts/evaluate_final.py, and every data-loading path here goes through
`data_ext.load_train_only_prepared`, which is built only on top of `src.data.load_train_only` (never
`load_dataset` / `load_test_only`). See `official_test_firewall.py` for the enforced guard and
`tests/test_research_extension.py::test_official_test_firewall_*` for the regression tests.
"""
