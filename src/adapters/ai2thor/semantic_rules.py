# semantic_rules.py

SEMANTIC_RECEPTACLE_RULES = {

    # 🍎 Food
    "Apple": ["Fridge", "Bowl", "Plate", "CounterTop", "DiningTable"],
    "Bread": ["Plate", "CounterTop", "Fridge"],
    "Egg": ["Fridge", "Bowl", "Pan", "Plate"],
    "Lettuce": ["Fridge", "Bowl", "Plate"],
    "Potato": ["Fridge", "Bowl", "CounterTop"],
    "Tomato": ["Fridge", "Bowl", "Plate"],

    # 🍽 Utensils
    "Knife": ["Drawer", "CounterTop"],
    "Fork": ["Drawer", "CounterTop"],
    "Spoon": ["Drawer", "CounterTop"],
    "Spatula": ["Drawer", "CounterTop"],
    "Ladle": ["Drawer", "CounterTop"],

    # ☕ Drinkware
    "Mug": ["Cabinet", "Shelf", "CounterTop"],
    "Cup": ["Cabinet", "Shelf", "CounterTop"],
    "Bottle": ["Fridge", "Cabinet", "CounterTop"],
    "WineBottle": ["Fridge", "Cabinet", "Shelf"],

    # 📚 Objects
    "Book": ["Desk", "Shelf", "Bed", "Sofa"],
    "Laptop": ["Desk", "Bed", "DiningTable"],
    "RemoteControl": ["Sofa", "CoffeeTable", "SideTable"],

    # 🛏 Bedroom
    "Pillow": ["Bed", "Sofa", "ArmChair"],
    "TeddyBear": ["Bed", "Sofa"],
    "Cloth": ["LaundryHamper", "Drawer"],

    # 🧴 Bathroom
    "SoapBar": ["SinkBasin", "Bathtub", "Shelf"],
    "ToiletPaper": ["ToiletPaperHanger"],
    "Towel": ["TowelHolder"],

    # 🧹 Cleaning
    "DishSponge": ["SinkBasin", "CounterTop"],
    "ScrubBrush": ["SinkBasin", "Shelf"],

    # 🖊 Office
    "Pen": ["Drawer", "Desk"],
    "Pencil": ["Drawer", "Desk"],

    # 🍳 Cookware
    "Pan": ["StoveBurner", "CounterTop"],
    "Pot": ["StoveBurner", "CounterTop"],
    "Kettle": ["StoveBurner", "CounterTop"],

    # 📦 Containers
    "KeyChain": ["Drawer", "Safe"],
    "CreditCard": ["Drawer", "Safe"],

}

def get_preferred_receptacles(obj_type):
    return SEMANTIC_RECEPTACLE_RULES.get(obj_type, [])

def build_semantic_map(controller):

    semantic_map = {}

    for obj in controller.last_event.metadata["objects"]:

        obj_type = obj["objectType"]

        parents = obj.get("parentReceptacles", [])

        if not parents:
            continue

        if obj_type not in semantic_map:
            semantic_map[obj_type] = set()

        for pid in parents:

            parent = next(
                o for o in controller.last_event.metadata["objects"]
                if o["objectId"] == pid
            )

            semantic_map[obj_type].add(parent["objectType"])
    print(semantic_map)

    return semantic_map
