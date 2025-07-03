import spacy
import re

class LinguisticsHelper():

    def __init__(self):
        """
        Initialize LinguisticsHelper.
        """
        # Load spaCy English model
        self._nlp = spacy.load("en_core_web_sm")

    def extract_named_entities_spacy(self, text: str) -> list[str]:
        """
        Extracts named entities (PERSON, ORG) from the input text using spaCy.
        Handles lowercase inputs gracefully.

        args:
            text (str): Text from which to extract named entities.

        returns (str): Text but only named entities.
        """
        doc = self._nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in {'PERSON', 'ORG'}]


    def is_coreferential(self, text: str) -> bool:
        """
        Detects whether the text contains third-person pronouns,
        suggesting it's a follow-up to a previous subject.

        args:
            text (str): Text to check for coreferences.
        
        returns (bool): Whether the specified text contains third-person pronouns.
        """
        pronouns = {
            'he', 'she', 'they', 'him', 'her', 'his', 'their', 'them', 'it',
            'himself', 'herself', 'themselves'
        }
        tokens = re.findall(r'\b\w+\b', text.lower())
        return any(token in pronouns for token in tokens)


    def should_carry_subjects(self, user_query: str, previous_subjects: list[str]) -> bool:
        """
        Decides whether to carry over the previous `_subjects_` to enrich a query.
        Criteria:
        - Use prior subjects only if:
            - The current query contains coreferential pronouns
            - AND does not introduce new named entities
        
        params:
            user_query (str): Human question.
            previous_subjects (str[]): Subjects of previous message.
        
        returns (bool): Whether subject carryover should occur according to relation
        between human question and previous subjects.
        """
        current_entities = [ent.lower() for ent in self.extract_named_entities_spacy(user_query)]
        previous_entities = [subj.lower() for subj in previous_subjects]

        has_coreference = self.is_coreferential(user_query)
        introduces_new_entity = any(ent not in previous_entities for ent in current_entities)

        return has_coreference and not introduces_new_entity