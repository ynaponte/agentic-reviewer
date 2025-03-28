import os

os.environ["OTEL_SDK_DISABLED"] = "True"

from src.article_writer.main import ArticleWriterFlow


flow = ArticleWriterFlow()

if __name__ == '__main__':
    flow.kickoff()