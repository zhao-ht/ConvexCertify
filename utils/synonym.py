def generate_synonym_list_by_dict(syn_dict, word):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''

    if not word in syn_dict:
        return [word]
    else:
        return syn_dict[word]