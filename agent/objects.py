from pydantic import BaseModel, Field, model_validator

class Answer(BaseModel):
    answer: str = Field(description="the answer that is needed for the user question")
    conversation: bool = Field(description="This is an indicator to know if the conversation should continue or not. If the agent has made the decision about refund this should be False. If conversation still need more talk then its True", default=True)

    @property
    def Answer(self) -> str:
        return self.answer


class PurchaseItem(BaseModel):
    item: str = Field(description="name of the item purchased")

    @property
    def name(self) -> str:
        return self.item

    @model_validator(mode="before")
    @classmethod
    def parse_input(cls, data):
        if isinstance(data, str):
            return {"item": data}
        if isinstance(data, dict) and "name" in data and "item" not in data:
            return {"item": data["name"]}
        return data
