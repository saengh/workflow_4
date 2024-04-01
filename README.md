# workflow_1

## m1_main:
Store all paths

## m2_xml_parser:
Extract title, abstract, iclm, claims, and CPC classes in separate columns
Combine title, abstract, and claims columns into CTB column

## power_query:
Using power query, add definitions to all CPC classes in the CPC classes column

## m3_assign_cpc_defs:
Merge CPC definitions into parsed_xml dataframe
Create another combined column CTB_CPC containg text from CTB and CPC class definitions

## m4_preprocessor:
Do conventional preprocessing to get word, phrase, and sentence tokens on all columns
Convert CTB and CTB_CPC word tokens back to sentences and generate TF-IDF document term matrix

## m5_nmf_topic_modeler:
Perform NMF modeling