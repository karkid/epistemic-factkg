CLAIM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",

    "required": [
        "id",
        "label",
        "claim",
        "claim_triples",
        "reasoning",
        "evidence",
        "context",
        "meta"
    ],

    "properties": {

        "id": {
            "type": "string"
        },

        "label": {
            "type": "string",
            "enum": ["supported", "refuted"]
        },

        "claim": {
            "type": "string"
        },

        "claim_triples": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "string"
                }
            }
        },

        "reasoning": {
            "type": "object",
            "required": ["structural"],
            "properties": {
                "structural": {
                    "type": "string",
                    "enum": [
                        "one-hop",
                        "conjunction",
                        "negation",
                        "multi-hop"
                    ]
                }
            }
        },

        "evidence": {
            "type": "object",

            "required": [
                "evidence_triples",
                "evidence_source",
                "evidence_source_type",
                "evidence_urls",
                "extract",
            ],

            "properties": {

                "evidence_triples": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "string"
                        }
                    }
                },

                "evidence_source": {
                    "type": "string"
                },

                "evidence_source_type": {
                    "type": "string"
                },

                "evidence_urls": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "uri"
                    }
                },
                "extract": {
                    "type": ["string", "null"]
                },
            }
        },

        "context": {
            "type": "object",

            "required": [
                "context_id",
                "context_type",
                "generator"
            ],

            "properties": {

                "context_id": {
                    "type": "string"
                },

                "context_type": {
                    "type": "string"
                },

                "generator": {
                    "type": "string"
                },

                "split": {
                    "type": ["string", "null"]
                }
            }
        },

        "meta": {
            "type": "object",

            "required": [
                "created_utc"
            ],

            "properties": {

                "created_utc": {
                    "type": "string",
                    "format": "date-time"
                },

                "notes": {
                    "type": ["string", "null"]
                }
            }
        }
    }
}
