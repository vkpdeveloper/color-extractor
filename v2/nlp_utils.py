import spacy
from spacy.pipeline import EntityRuler


class QueryProcessor:
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Fallback if model loading fails (though we just installed it)
            # This handles cases where the link/name might differ slightly in some envs
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()

        self._add_fashion_patterns()

    def _add_fashion_patterns(self):
        # Create EntityRuler and add it before the 'ner' component (or at start if 'ner' missing)
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        patterns = [
            # GENDER
            {"label": "GENDER", "pattern": [{"LOWER": "men"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "man"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "male"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "boys"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "boy"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "women"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "woman"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "female"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "girls"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "girl"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "ladies"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "lady"}]},
            {"label": "GENDER", "pattern": [{"LOWER": "unisex"}]},

            # MATERIAL
            {"label": "MATERIAL", "pattern": [{"LOWER": "cotton"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "denim"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "leather"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "wool"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "silk"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "polyester"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "linen"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "nylon"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "velvet"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "satin"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "canvas"}]},
            {"label": "MATERIAL", "pattern": [{"LOWER": "fleece"}]},

            # COLOR (Basic Set)
            {"label": "COLOR", "pattern": [{"LOWER": "red"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "blue"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "green"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "black"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "white"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "yellow"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "pink"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "baby pink"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "purple"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "orange"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "grey"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "gray"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "brown"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "beige"}]},
            {"label": "COLOR", "pattern": [{"LOWER": "navy"}]},

            # CATEGORY (Basic Set - can be expanded)
            {"label": "CATEGORY", "pattern": [{"LOWER": "shirt"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "t-shirt"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "tshirt"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "tee"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "jeans"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "pants"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "trousers"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "shorts"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "dress"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "skirt"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "jacket"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "coat"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "hoodie"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "sweater"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "sweatshirt"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "cardigan"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "blazer"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "kurta"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "kurti"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "saree"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "sari"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "lehenga"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "shoes"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "sneakers"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "boots"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "sandals"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "heels"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "flats"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "loafers"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "tops"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "top"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "polo"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "blouse"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "leggings"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "joggers"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "trackpants"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "chinos"}]},

            # ONE-PIECE / OTHER (Outerwear only)
            {"label": "CATEGORY", "pattern": [{"LOWER": "jumpsuit"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "dungarees"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "romper"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "tunic"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "kaftan"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "shrug"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "waistcoat"}]},
            {"label": "CATEGORY", "pattern": [{"LOWER": "suit"}]},

            # PATTERN
            {"label": "PATTERN", "pattern": [{"LOWER": "solid"}]},
            {"label": "PATTERN", "pattern": [{"LOWER": "striped"}]},
            {"label": "PATTERN", "pattern": [{"LOWER": "checked"}]},
            {"label": "PATTERN", "pattern": [{"LOWER": "printed"}]},
            {"label": "PATTERN", "pattern": [{"LOWER": "textured"}]},

            # BRAND
            {"label": "BRAND", "pattern": [{"LOWER": "adidas"}]},
            {"label": "BRAND", "pattern": [{"LOWER": "damensch"}]},
            {"label": "BRAND", "pattern": [
                {"LOWER": "allen"}, {"LOWER": "solly"}]},
            {"label": "BRAND", "pattern": [
                {"LOWER": "rare"}, {"LOWER": "rabbit"}]},
            {"label": "BRAND", "pattern": [{"LOWER": "marks"}, {
                "LOWER": "&"}, {"LOWER": "spencer"}]},
            {"label": "BRAND", "pattern": [{"LOWER": "marks"}, {
                "LOWER": "and"}, {"LOWER": "spencer"}]},
            {"label": "BRAND", "pattern": [{"LOWER": "m&s"}]},
        ]

        ruler.add_patterns(patterns)

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            label = ent.label_
            text_val = ent.text.lower()

            if label not in entities:
                entities[label] = []
            if text_val not in entities[label]:
                entities[label].append(text_val)

        return entities

    def build_meili_filter(self, entities):
        """
        Constructs a Meilisearch filter string from extracted entities.
        Assumption: We are filtering on 'title', 'description', or 'attributes.value'.

        Refined Strategy:
        1. If GENDER is found, we filter Title OR Description OR Attributes.
        2. If MATERIAL is found, we filter Title OR Description OR Attributes.
        3. If COLOR is found, we filter Title OR Description OR Attributes.
        4. If PATTERN is found, we filter Title OR Attributes (Pattern usually not reliable in desc alone, but we can add description if desired. Let's stick to Title/Attr for Pattern as previously set, or consistency: Title/Desc/Attr).
           User asked to check "each other filter inside description and also inside our attributes value".
           So we will apply Title OR Description OR Attributes to ALL.
        """
        filters = []

        # Helper to build OR clause for a single term across fields
        def build_field_or(term):
            return f'(title CONTAINS "{term}" OR description CONTAINS "{term}" OR attributes.value CONTAINS "{term}")'

        # Helper to build NOT clause for a single term across fields
        def build_field_not(term):
            return f'(NOT title CONTAINS "{term}" AND NOT description CONTAINS "{term}" AND NOT attributes.value CONTAINS "{term}")'

        # GENDER FILTER
        if "GENDER" in entities:
            gender_conditions = []

            # Expansions: term -> list of variations
            expansions = {
                "men": ["men", "man", "men's"],
                "man": ["men", "man", "men's"],
                "male": ["male"],
                "women": ["women", "woman", "women's"],
                "woman": ["women", "woman", "women's"],
                "female": ["female"],
                "boy": ["boy", "boys"],
                "boys": ["boy", "boys"],
                "girl": ["girl", "girls"],
                "girls": ["girl", "girls"]
            }

            # Exclusions: term -> list of terms to exclude (because term is substring of excluded)
            exclusions = {
                "men": ["women", "woman", "female"],
                "man": ["women", "woman", "female"],
                "men's": ["women", "woman", "female"],
                "male": ["female"],
            }

            processed_terms = set()
            for g in entities["GENDER"]:
                # Expand specific terms
                terms_to_check = expansions.get(g, [g])

                for term in terms_to_check:
                    if term in processed_terms:
                        continue
                    processed_terms.add(term)

                    clause = build_field_or(term)

                    # Apply exclusions
                    if term in exclusions:
                        exclude_list = exclusions[term]
                        not_conditions = [build_field_not(
                            ex) for ex in exclude_list]
                        not_clause = " AND ".join(not_conditions)
                        clause = f'({clause} AND {not_clause})'

                    gender_conditions.append(clause)

            if gender_conditions:
                filters.append(f"({' OR '.join(gender_conditions)})")

        if "MATERIAL" in entities:
            material_conditions = [build_field_or(
                m) for m in entities["MATERIAL"]]
            filters.append(f"({' OR '.join(material_conditions)})")

        if "COLOR" in entities:
            color_conditions = [build_field_or(c) for c in entities["COLOR"]]
            filters.append(f"({' OR '.join(color_conditions)})")

        if "PATTERN" in entities:
            # Pattern logic updated to include description as well
            pattern_conditions = [build_field_or(
                p) for p in entities["PATTERN"]]
            filters.append(f"({' OR '.join(pattern_conditions)})")

        if "CATEGORY" in entities:
            # Helper to include 'category' field specifically for CATEGORY entities
            def build_category_or(term):
                return f'(title CONTAINS "{term}" OR description CONTAINS "{term}" OR attributes.value CONTAINS "{term}" OR category CONTAINS "{term}")'

            category_conditions = [build_category_or(
                c) for c in entities["CATEGORY"]]
            filters.append(f"({' OR '.join(category_conditions)})")

        if "BRAND" in entities:
            # Brand filter: brand CONTAINS "term"
            brand_conditions = [f'brand CONTAINS "{
                b}"' for b in entities["BRAND"]]
            filters.append(f"({' OR '.join(brand_conditions)})")

        if not filters:
            return None

        return " AND ".join(filters)


# Singleton instance for easy import
# Initialize later to avoid import-time costs if not needed immediately
_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = QueryProcessor()
    return _processor
