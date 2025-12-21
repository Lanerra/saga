"""Build parameterized Cypher statements for `data_access`.

Notes:
    Security:
        Cypher identifiers (for example, labels and relationship types) are not uniformly
        parameterizable in Neo4j. Builders must treat any identifier-like inputs as
        untrusted unless they come from application-controlled allowlists, and they must
        validate or constrain such inputs before inserting them into query text or passing
        them to procedures that create typed relationships.
"""
