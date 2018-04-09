#  notes on multi mls

- implemented the increment internal function for multi mls, check wether it does the right thing
- implemented the multimlsdim class, check wether it does the right thing
- seems like i implemented the add dimension functionality, needs to be checked
- seems like i  implemented the index constructor for multi mls types, needs to be checked
- seems like i rewrote all the pitch  in the manner n00_1 -> -1 n00_2 -> -2 ...  keept the dim number interface, needs to be tested
- seems like i rewrote the whole liouville thing, needs to be tested

- then for the whole output thing one sanity check that the mls dim objects refer to the same mls type and generalized n00 functionality
- need to compile and debug
- need to write a first complete example

- maybe do some clean up work after everything is tested, like the pitch functions could be shortened
- after that rewrite all the pitch functions etc.(see below) and rewrite all the instances where they occur, this could then be the whole thing, actually
- then there needs to be functionality for different expecation values and so forth

- need to implement sanity check so that the mls dim setup does not mix for different types of mls!
- need to rewrite all index routines that get integer dimension numbers into routines that take dim class and child class objects. that is safer for the multi-mls and it is in good agreement with oo style
- the whole n00 functionality is affected by the multimls stuff, needs to be implemented
- test what happens when there are no mls at all, just modes
- check the type cast things in the MultiMLSDim constructors
- check the type casting in MLSDim::IsEqual(Dim * input) and other functions, that looks like a segfault!
- check the index constructor "Before truncation: setup stage" step, it is wrong but it works anyway so it seems to be obsolete
- first dimension cannot be truncated but all other seem to be truncateable also with the multi mls stuff, check that
- search for all TODO marked code lines and do the thing...
- what happens with the superradiant stuff?

