# Dataset access and privacy

Obtain `data.zip` from your authorized ProjectPro project download page and extract
`Canva_reviews.xlsx` locally. Supply its path with `--data`; do not commit it.

The inspected file has 1,500 rows and 13 columns. Only `review` and `Sentiment`
are used. Labels: 1,032 Positive and 468 Negative. Other columns include reviewer
names, IDs, profile-image URLs and app-reply metadata; these are unnecessary for
the model and have not been copied into this repository.

The methodology calls these publicly available app reviews; the supplied files
do not establish original collection dates, sampling, labeling protocol or a
redistribution license. Public availability is not a redistribution grant.
No raw review excerpts or original notebook outputs are published here.

For anyone without dataset access, `python -m unittest discover -s tests -v`
runs offline with authored synthetic examples. These examples are tests, not the
real evaluation dataset. The recorded real-data metrics cannot be independently
reproduced without lawful access to the same input file.
