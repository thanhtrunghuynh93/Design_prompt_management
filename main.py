import pandas as pd
import os
from dotenv import load_dotenv
from tools.metadata_extractor import describe_image_with_langchain, tagging_image_with_langchain
import time

load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

df = pd.read_csv("data.csv")
# Initialize columns
df["description"] = ""
df["tags"] = ""

max_num_try = 3
num_items = len(df)

for i in range(num_items):
    print(i)
    for attempt in range(max_num_try):
        try:
            description = describe_image_with_langchain(
                llm, df["MK"][i], local_img=False
            )
            tags = tagging_image_with_langchain(
                llm, df["MK"][i], local_img=False
            )

            # Save results to dataframe
            df.at[i, "description"] = description
            df.at[i, "tags"] = tags
            break 
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for item {i}: {e}")
            time.sleep(60)  # Wait before retrying
            if attempt == max_num_try - 1:
                print(f"Skipping item {i} after {max_num_try} tries.")
            

    # Save the updated dataframe to a new CSV file
    if i % 10 == 0:  # Save every 10 items
        df.to_csv("data_tagged.csv")
        time.sleep(60)  # To avoid hitting rate limits

df.to_csv("data_tagged.csv")