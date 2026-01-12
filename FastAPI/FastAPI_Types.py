import datetime
from typing import List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI()
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    tags: List[str] = []
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
@app.post("/items/", response_model=Item)
async def create_item(item: Item) -> Item:
    return item
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int) -> Item:
    sample_item = Item(
        id=item_id,
        name="Sample Item",
        description="This is a sample item",
        price=10.5,
        tax=1.5,
        tags=["sample", "item"]
    )
    return sample_item
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item) -> Item:
    updated_item = item.copy(update={"id": item_id})
    return updated_item
@app.get("/items/", response_model=List[Item])
async def list_items() -> List[Item]:  
    items = [
        Item(
            id=1,
            name="Item 1",
            description="First item",
            price=20.0,
            tax=2.0,
            tags=["first", "item"]
        ),
        Item(
            id=2,
            name="Item 2",
            description="Second item",
            price=30.0,
            tax=3.0,
            tags=["second", "item"]
        )
    ]
    return items 
@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(item_id: int) -> dict:
    return {"message": f"Item with id {item_id} has been deleted"}  
@app.get("/items/search/", response_model=List[Item])
async def search_items(name: Optional[str] = None, max_price: Optional[float] = None) -> List[Item]:
    all_items = [
        Item(
            id=1,
            name="Item 1",
            description="First item",
            price=20.0,
            tax=2.0,
            tags=["first", "item"]
        ),
        Item(
            id=2,
            name="Item 2",
            description="Second item",
            price=30.0,
            tax=3.0,
            tags=["second", "item"]
        ),
        Item(
            id=3,
            name="Special Item",
            description="A special item",
            price=25.0,
            tax=2.5,
            tags=["special", "item"]
        )
    ]
    filtered_items = [
        item for item in all_items
        if (name is None or name.lower() in item.name.lower()) and
           (max_price is None or item.price <= max_price)
    ]
    return filtered_items
@app.get("/items/summary/", response_model=dict)
async def items_summary() -> dict:
    summary = {
        "total_items": 3,
        "average_price": 25.0,
        "most_expensive_item": {
            "id": 2,
            "name": "Item 2",
            "price": 30.0
        }
    }
    return summary
@app.patch("/items/{item_id}", response_model=Item)
async def partial_update_item(item_id: int, item: Item) -> Item:
    updated_item = item.copy(update={"id": item_id})
    return updated_item
@app.get("/items/{item_id}/details/", response_model=Union[Item, dict])
async def get_item_details(item_id: int) -> Union[Item, dict]:
    if item_id == 1:
        return Item(
            id=item_id,
            name="Item 1",
            description="First item",
            price=20.0,
            tax=2.0,
            tags=["first", "item"]
        )
    else:
        return {"message": f"No detailed information available for item with id {item_id}"} 

@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to the FastAPI Types Example!"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    