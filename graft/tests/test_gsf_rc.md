# tests/test_gsf_rc.py

- _gene_gsf · function · L5-L9 — def _gene_gsf(gsf, tx, strand="+")
- test_rc_is_an_involution · function · L12-L15 — def test_rc_is_an_involution(gsf)
- test_rc_mirrors_coordinates_and_flips_strand · function · L18-L23 — def test_rc_mirrors_coordinates_and_flips_strand(gsf)
- test_rc_recomputes_phases_five_to_three_prime · function · L26-L38 — def test_rc_recomputes_phases_five_to_three_prime(gsf): # CDS lengths 100 and 101: forward phases A (0) then C (2). RC flips the strand but not the # transcript: the 100-bp piece is still read first (A) and the 101-bp piece keeps phase # (3 - 100 mod 3) mod 3 = 2 (C). In the coordinate-sorted feature list the mirrored 101-bp piece # now comes first, so the list reads C then A; a wrong phase (e.g. 1 -> B) must not appear.
- test_rc_keeps_utrs_in_transcription_order_and_is_canonical · function · L41-L45 — def test_rc_keeps_utrs_in_transcription_order_and_is_canonical(gsf)
- test_rc_involution_for_five_prime_partial_model · function · L48-L56 — def test_rc_involution_for_five_prime_partial_model(gsf): # first CDS starts in phase 2 (5'-partial model): RC must keep that phase and stay an involution
