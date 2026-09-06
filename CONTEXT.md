# Fantasy Premier League Manager

Squad selection and chip planning across a Fantasy Premier League season.

## Language

**Gameweek**: The numbered scoring period to which fixtures belong. A team may have zero, one or multiple fixtures in a Gameweek.

**Completed Gameweek**: A Gameweek in which every scheduled fixture has finished playing. Its scores may still await official review.

**Finalized Gameweek**: A completed Gameweek whose fixtures have all passed official score review.
_Avoid_: Finished Gameweek when official review is intended.

**Chip decision**: The choice of at most one chip for a Gameweek, together with its consequences for free transfers.

**Pending chip**: A chip selected for an upcoming Gameweek whose choice can still be revised before the deadline.
_Avoid_: Spent chip for a revisable choice.

**Free-transfer bank**: The accumulated free transfers available to change the squad without a points deduction.

**Season replay**: A simulation of squad decisions and scoring over historical Gameweeks. Its chip timing can account for the simulation's end date.
