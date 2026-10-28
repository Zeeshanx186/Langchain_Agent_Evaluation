from langchain.tools import tool
from data import users_data

from dotenv import load_dotenv
import logging
from objects import PurchaseItem
from datetime import date

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@tool
def get_user_purchase_history(user: str) -> dict:
    """
    This tool takes the user's username and return dictionary of the user purchase history if the user exists.

    """
    logging.info(f"using get_user_purchase_history tool on users: {user}")
    if users_data.get(user):
        return users_data[user]
    return {"status": "User Not Found"}
       
@tool
def calculate_right_refund(user: str, items: list[PurchaseItem]) -> str:
    """
    This tool takes a list of the items to be returned and the username. It will return how much the refund is worth.
    """
    logging.info(f"using calculate_right_refund tool to get user's: {user} refund")

    if users_data.get(user):
        total_refund = 0
        user_items = {k.lower().replace("_", " "): v["price"] for k, v in users_data[user].items()}

        for item in items:
            key = item.name.lower().replace("_", " ")
            if key in user_items:
                total_refund += user_items[key]
        return f"the value of the refund is : {total_refund}"
    return {"status":"there was no username found"}

@tool
def issue_refund_ticket(user:str, refund_amount:int)->str:
    """this tool will send a ticket to the right team to return the refund"""

    logging.info(f"using issue_refund_ticket tool for user: {user}")
    return f"a refund ticket has been issued successfully with the amount of {refund_amount}"

@tool
def get_today_date() -> str:
    """this tool will get the current date"""

    logging.info("using get_today_date tool")
    return date.today().strftime("%Y-%m-%d")
    

# print(get_user_purchase_history.invoke("user123"))
# print(calculate_right_refund.invoke({"user": "user123", "items": ["book", "pen"]}))
#print(issue_refund_ticket.invoke({"user": "user123", "refund_amount": 30}))
# print(get_today_date.invoke("user123"))