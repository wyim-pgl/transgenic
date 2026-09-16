"""C3 established primitives only; genomic phase projection is BLOCKED.

Workplan §2.6 requires frameshift-aware projection across introns mod 3,
six phase{0,1,2} x strand{+,-} channels and route-A disagreement -> zero
support. It does not pin the phase-0 anchor, overlapping/inconsistent-HSP
resolution or the RICH17 HSP/BTOP semantics. Do not substitute a guessed
projector. No BAM ingestion, molecule aggregation or raster is implemented.
"""


def genomic_strand(flag, ts):
    """Pure FLAG16 x read-relative ts:A composition; not an alignment filter.

    Authority: .atl/quarantine.md §1m, ts:A/FLAG16 row and
    revision/scripts/67_orientation_audit.py:238-240 at base 950cf4c.
    """
    if type(flag) is not int or flag < 0 or ts not in ('+', '-'):
        raise ValueError('nonnegative SAM flag and explicit ts +/- required')
    return ('-' if ts == '+' else '+') if flag & 16 else ts


def reconcile_phases(route_a, route_b):
    """Compare already-labelled phases for the SAME base/strand/protein pair.

    None means no support (all phase-channel values zero), NOT phase label 0.
    Returns (A, B, disagreement). Does not resolve multilabel observations.
    """
    for phase in (route_a, route_b):
        if phase is not None and (type(phase) is not int or phase not in (0, 1, 2)):
            raise ValueError('phase must be 0, 1, 2 or absent')
    if route_a is not None and route_b is not None and route_a != route_b:
        return None, None, True
    return route_a, route_b, False


def project():
    raise ValueError('UNRESOLVED phase anchor/HSP/BTOP semantics; production blocked by A46')


if __name__ == '__main__':
    project()
