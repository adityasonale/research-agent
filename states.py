from typing_extensions import TypedDict, NotRequired, Optional

class ResearchState(TypedDict):
    # ---- Required at initialisation ----
    query:          str
    iteration:      int
    max_iterations: int
    notes:          list[str]

    # ---- Populated during run ----
    research_plan:  NotRequired[Optional[list[str]]]
    search_plan:    NotRequired[Optional[list[str]]]
    search_query:   NotRequired[Optional[str]]
    decision:       NotRequired[str]