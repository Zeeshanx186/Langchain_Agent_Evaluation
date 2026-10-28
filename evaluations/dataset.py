from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall


dataset = EvaluationDataset(
    goldens = [
        Golden(
            input = "I am user123, I would like refund for the Book and the pen",
            expected_output = "A refund ticket for 30$ has been issued successfully",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history"),
                ToolCall(name = "calculate_right_refund"),
                ToolCall(name = "issue_refund_ticket"),
                ToolCall(name = "get_today_date")
            ]
        ),
        Golden(
            input = "I am user123, I would like refund for the Book",
            expected_output = "A refund ticket for 23$ has been issued successfully",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history"),
                ToolCall(name = "calculate_right_refund"),
                ToolCall(name = "issue_refund_ticket"),
                ToolCall(name = "get_today_date")
            ]
        ),
        Golden(
            input = "I am user123, I would like no refund, I just want to know what I purchased.",
            expected_output = "No refund tickets were issued only a list of what they purchased",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history")
            ]
        ),
        Golden(
            input = "I am user999, I would like refund for the Notebook",
            expected_output = "Tell the user that the username is not valid and ask for username again before giving info about the refund",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history")
            ]
        ),
        Golden(
            input = "I am user126, I would like refund for the water bottle",
            expected_output = "no refund given because it is outside the 30 days window",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history"),
                ToolCall(name = "get_today_date")
            ]
        ),
        Golden(
            input = "I am user126, I would like refund for the water bottle and backpack",
            expected_output = "A refund ticket if 45$ has been issued successfully only for the backpack but the water bottle is  not elligible for refund because it outside the 30 days refund policy.",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history"),
                ToolCall(name = "get_today_date"),
                ToolCall(name = "calculate_right_refund"),
                ToolCall(name = "issue_refund_ticket")
            ]
        ),
        Golden(
            input = "I am user126, I would like refund for sneakers",
            expected_output = "No refund was given, because you did not purchase this item.",
            expected_tools = [
                ToolCall(name = "get_user_purchase_history")
            ]
        ),
        

    ]
)