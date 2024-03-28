# workflow_1

## m1_main:
# Store all paths

## m2_xml_parser:
## Extract title, abstract, iclm, claims, and CPC classes in separate columns
## Combine title, abstract, and claims columns into CTB column

## power_query:
## Using power query, add definitions to all CPC classes in the CPC classes column

## m3_assign_cpc_defs:
## Merge CPC definitions into parsed_xml dataframe
## Create another combined column CTB_CPC containg text from CTB and CPC class definitions

## m4_preprocessor:
## Do conventional preprocessing to get word, phrase, and sentence tokens on all columns
## In subsequent modules, both word and phrase tokens are used independently

## m5_lda_preprocessor:
## Perform LDA modeling on by treating each row as an entire dataset in itself and each column (title, abstract, iclm, and claims) as a separate document
## LDA should give topic names for each document which would serve as tokens for next steps

## m6_document_token_matrix:
## Convert dataframe to a matrix where rows are document indices, columns are unique topics by LDA from entire dataset, and probabilities of topics are values in matrix

## m7_jsd:
## Find similarity between pair of documents based on similarity between their probability similarities

## m8_clustering:
## Perform clustering using HDBSCAN