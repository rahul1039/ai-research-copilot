def extract_regression_models(all_chunks):
    context = " ".join([doc.page_content for doc in all_chunks])