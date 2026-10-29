"""Addenda only. The complete existing prompts remain unchanged prefixes."""

DETECTION_ADDENDUM = '''

ADDITIONAL FABRICATION REFERENCE RULES:
The assembled-component rule above is extended to a clearly evidenced exploded
assembly: use assembly_candidate with assembly_basis=exploded_assembly only when
visible assembly axes, explicit installation steps or connection features show
how TWO OR MORE distinct parts form this joint. Give concrete visible evidence.
Separate unrelated profile drawings, tools and catalogue tables remain non-assembly.
An exploded representation is not evidence that its loose parts are already fitted;
retain uncertainty about component membership. Never infer hidden connections.

Do not group views during detection. Instead record reference_marks on each
individual region when visible: kind=enlargement, section, variant, assembly_step
or alternative_code; role=source for a marker locating a joint, caption for the
corresponding view caption or contextual label. Keep exact text, marker token,
complete normalized box and visible evidence. Optional leader_points are the
actual reference pointer/cutting trace from the source marker towards the joint,
not a guessed path. Empty reference_marks is valid when no such marks are visible.
Examples include X at a joint and X 2:1 above its enlargement; A/B cutting markers
and A-A/B-B section captions; Standard, HI and SI variant labels; numbered assembly
steps; and explicitly scoped alternative part numbers in parentheses. Do not
invent a counterpart, resolve membership, or turn these marks into component IDs.
Retain their complete source text in box, separately from context_annotations:
context_annotations still enumerates component leaders only. A bare repeated X,
similar geometry, shared title, or repeated part number alone proves no relation.
Standard/HI/SI are distinct variants, not extra language translations. Do not merge
them into one assembly. Different source languages for one callout do not create
additional physical components. Numbers identifying steps, cutting references,
dimensions, tool operations and standards such as EN 10025 are not profile IDs.
Use REVIEW for ambiguous pointers, missing captions or overlapping scope. Preserve
every independent view and all existing crop/coverage requirements above.
'''

COVERAGE_ADDENDUM = '''

ADDITIONAL FABRICATION AUDIT:
The eligible-assembly rule also admits exploded_assembly when visible assembly
axes, connection features or explicit installation steps establish a multi-part
joint; a group of unrelated loose profiles or tools still does not qualify.
Check source reference markings, enlargements and section captions without merging
their views. In every replacement/addition retain any visible reference_marks with
exact text, marker, box, role, kind and supporting observations. Mark uncertainty
instead of inventing missing X/section counterparts or variant scope. Keep
Standard/HI/SI alternatives and assembly steps distinguishable. These references
do not supply component counts or authorize transferring codes from a table.
'''

GROUPING_ADDENDUM = '''

ADDITIONAL FABRICATION RELATIONS:
Supplied source_reference_hints describe original reference marks, not confirmed
membership. A joint locator X and its X 2:1 enlargement, or a cutting marker A and
its A-A section, may propose a directed source-view relation only when the actual
marker/caption and its scope are visible. Explain that exact evidence. The same
letter or part number by itself is insufficient. Preserve ambiguous matches.
Keep Standard, HI and SI variants separate even when their drawings are similar.
Keep assembly step order, optional accessories, tool diagrams, reference tables
and parenthesized alternative-code legends as context; none automatically merges
views or identifies an additional physical component. Several translated labels
for the same object are not several objects. Keep the existing response schema:
model groups remain proposals; source corroboration decides primary membership.
'''

CALLOUT_ADDENDUM = '''

ADDITIONAL FABRICATION CALLOUT RULES:
Distinguish component leaders from section/cutting traces, enlargement pointers,
assembly motion arrows, numbered installation steps, tools and variant markers.
X, A-A/B-B, E1/E2 when used as view references, Standard/HI/SI and a standard number
such as EN 10025 are not component identifiers merely because they are nearby.
Read actual component callouts normally; do not classify solely from a token.
Multiple source-language translations in ONE physical callout block sharing ONE
leader remain one occurrence. For explicitly printed alternative identifiers in
that same block, preserve their complete literal identifier reading, including
parentheses and line order, in that occurrence. Do not invent another occurrence
without a distinct source leader, and do not select or transfer a code from a
nearby legend/table without source ownership. Alternative conditions stay visible
in source evidence. Different leaders with identical codes remain separate.
Names alone and codes alone remain valid under the complete original rules above.
'''

NAME_ADDENDUM = '''

ADDITIONAL FABRICATION NAME RULES:
For component_name, retain only literal words in the associated source callout.
Several translations in one callout do not create new components. Do not translate
or infer a missing name from a code, shape, variant, installation step, reference
table or neighboring view. Standard/HI/SI and section/enlargement markers describe
scope unless that source callout explicitly names a component. Preserve explicit
optional/alternative wording in source observations; never spread one variant's
name to another. The existing allowed fields and evidence requirements stay in force.
'''

DETECTION_ADDENDUM += '''

ADDITIONAL GENERAL CATALOG CASES:
Native text can contain prose while bold article codes and captions are vector
outlines. Inspect the pixels even when native spans exist; absence from spans is
not absence from the page. Retain full labels, faint layers, remote leaders and
local component legends in the context, without borrowing adjacent joint labels.
Use reference_marks kind=component_legend for a visible local numbered key needed
to interpret component leaders. Retain the complete key rows and scoped heading.
Use kind=detail_locator for numbered/lettered circles on an overview linked to
explicit Detail captions. Such overview-to-joint relations are CONTEXT only:
separate different joints, and never classify locator numbers as profile codes.
They differ from component item markers and numbered assembly instructions.
Classify each drawing from visible physical connections: a Profile summary may
contain an assembled sleeve, reinforcement or connector as well as single
extrusions. Cavities, material hatching and dimensions alone are not extra parts.
An exploded kit is eligible only with explicit assembly evidence, not just many
nearby catalogue items. Record exact visible variant captions using kind=variant,
marker=text=the complete literal caption. Variants are not limited to HI/SI:
opening mode, handedness, mounting method and dimensional option can delimit
different scopes. Do not infer unprinted scope or combine mutually exclusive or
OPTION accessories. A plus sign can describe a printed kit, not extra leaders.
'''

COVERAGE_ADDENDUM += '''

Inspect outlined codes even on pages with reliable native prose. Check complete
numbered legend rows, separate locator circles from component leaders, and do not
exclude all drawings under Profile summary. Retain full remote labels and each
independent leader from a shared label. Never make a faint layer disappear by
cropping it or by treating visible small parts as background without evidence.
'''

CALLOUT_ADDENDUM += '''

ADDITIONAL LOCAL LEGEND AND MULTI-LEADER CONTRACT:
One shared physical label can have SEVERAL independent complete leaders to
different components. Return one observation per independent leader, reusing the
exact same label wording/box; repeat readings of the SAME leader are not new
observations. A Y branch with a shared physical stem, crossings, assembly motion
axes and uncertain ownership remain REVIEW. Do not count visible bolts or shapes.
Inspect vector-outlined codes even when supplied native candidates omit them.
For a numeric/letter component marker whose actual code/name is in a clearly
scoped local key, return category=item_marker and raw_text=the literal marker at
the leader, NOT its resolved code. Supply legend_link only when that key applies
explicitly to this joint. A source line actually pointing to a physical component
uses leader_kind=component even though its printed label is category=item_marker.
Supply the link with
exact marker, code, literal optional name, full row_text,
row_box/code_box/name_box, scope_text and scope_box enclosing BOTH marker and key,
and concrete source evidence. Null when ambiguous. Do not borrow rows from a
neighboring variant or infer an association from proximity. The independent audit
must read the source again and independently supply the same link. Scope/quote
conflicts must remain unresolved. These item markers differ from Detail locator
numbers and assembly instructions; pure numeric article codes can still be real.
Mutually exclusive or/OPTION choices stay separate context. Do not create another
physical occurrence from a code list or a quantity field without another leader.
'''

GROUPING_ADDENDUM += '''

Numbered Detail locator circles identify different source joints and provide
context only. Never union all joints belonging to the same elevation or roof.
Honor exact printed variant captions beyond HI/SI: handedness, opening mode,
mounting choice and dimension range delimit variants. Similar geometry, repeated
title, shared kit code, or or/OPTION alternatives do not merge variants.
'''

NAME_ADDENDUM += '''

A supported numbered local legend is an explicitly associated source callout:
its literal name may be reused only with the supplied source-resolved legend
evidence. Do not guess unresolved marker meanings. Scoped cross-page table names
are assigned separately from independently audited OCR, never from model memory
or shape; do not fabricate those names during this local crop reading.
'''

OCR_LAYOUT_ADDENDUM = '''

ADDITIONAL CATALOG TEXT/SCOPE RULES:
Bold part codes and Detail captions can be vector outlines while prose is native
text. Include every such text region. The existence of native spans never proves
that all codes are represented. Keep numbered keys and variant captions readable.
scope contains only explicitly PRINTED scope labels: kind=system for the printed
system designation, section for its printed product family/section, variant for
the exact printed local option; manufacturer and edition for those printed labels
when available. Section is the product-family namespace repeated in catalog
headers, not a generic table group such as Profiles, Accessories or Tools.
Give literal text and box; never infer from a file
name, source page number, visual similarity or a neighboring page. Page-level
scope applies only if explicitly global. A region can have its own scope labels
when explicit bounded columns/headings establish that scope. Otherwise use [].
Separate alternative kits/options as text scopes where source boundaries allow;
never synthesize a common variant label. Product tables must retain exact code
and name in their original row/columns, including blank cells and conditions.
'''

OCR_READ_ADDENDUM = '''

Inspect native and outlined glyphs in the same region. Transcribe item-marker
keys, article identifiers, names, conditions, or/OPTION alternatives and literal
scope headings without creating extra rows. Quantity, cutting formula, dimensions
and tool references stay in their printed columns. Scope is never inferred.
'''
