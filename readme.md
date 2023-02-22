# NDSH-MetaSearch

A simple tool performs semantic search for the nerc metadata.

# What did I do?

1. parse the metadata into a `DataFrame` with the following columns, using the `DataSourceBuilder` class:

    - `recordId`
    - `title`
    - `identifier`
    - `abstract`
    - `hierarchy_level`

2. clean the `abstract`/`title` columns. original column is renamed as `raw_abstract`/`raw_title`. Cleanning involves the following:

    1. all lower case
    2. remove line break e.g. `\t`, `\n`
    3. remove non-acsii-char
    4. fix_unicode errors

3. Using the pre-trained model: `multi-qa-MiniLM-L6-cos-v1` from `sbert` to embed the `abstract`/`title` columns, and save the embedding as `abstract_emb`/`title_emb` columns.

4. `Searcher` class take the `DataFrame` as input, performs the semantic search. it embedded the query on fly, and use the `cosine_similarity` to find the `top k` similar records.

5. build a `FastApi` app to serve the search result.