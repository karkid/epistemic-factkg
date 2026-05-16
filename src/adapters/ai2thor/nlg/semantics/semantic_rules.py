from typing import Dict, List

from src.infra.rdf.formatter import ai2thor_object_type_from_entity_id

SEMANTIC_RECEPTACLE_RULES = {
    # 🍎 Food
    "Apple": ["Fridge", "Bowl", "Plate", "CounterTop", "DiningTable"],
    "Bread": ["Plate", "CounterTop", "Fridge"],
    "Egg": ["Fridge", "Bowl", "Pan", "Plate"],
    "Lettuce": ["Fridge", "Bowl", "Plate"],
    "Potato": ["Fridge", "Bowl", "CounterTop"],
    "Tomato": ["Fridge", "Bowl", "Plate"],
    # 🍽 Utensils & Kitchenware
    "Knife": ["Drawer", "CounterTop"],
    "ButterKnife": ["Drawer", "CounterTop"],
    "Fork": ["Drawer", "CounterTop"],
    "Spoon": ["Drawer", "CounterTop"],
    "Spatula": ["Drawer", "CounterTop"],
    "Ladle": ["Drawer", "CounterTop"],
    "Bowl": ["Cabinet", "CounterTop", "DiningTable"],
    "Plate": ["Cabinet", "CounterTop", "DiningTable"],
    # ☕ Drinkware
    "Mug": ["Cabinet", "Shelf", "CounterTop"],
    "Cup": ["Cabinet", "Shelf", "CounterTop"],
    "Bottle": ["Fridge", "Cabinet", "CounterTop"],
    "WineBottle": ["Fridge", "Cabinet", "Shelf"],
    # 🧂 Seasonings & Small Items
    "SaltShaker": ["Cabinet", "CounterTop", "DiningTable"],
    "PepperShaker": ["Cabinet", "CounterTop", "DiningTable"],
    # 🍳 Cookware & Appliances
    "Pan": ["StoveBurner", "CounterTop", "Cabinet"],
    "Pot": ["StoveBurner", "CounterTop", "Cabinet"],
    "Kettle": ["StoveBurner", "CounterTop"],
    # 🌿 Decorative
    "HousePlant": ["SideTable", "CounterTop", "Shelf", "Floor"],
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
    # 📦 Containers
    "KeyChain": ["Drawer", "Safe"],
    "CreditCard": ["Drawer", "Safe"],
}


def get_preferred_receptacles(obj_type: str) -> List[str]:
    ai2thor_type = ai2thor_object_type_from_entity_id(obj_type)
    obj_type_lower = ai2thor_type.lower()
    # print(f"Looking up preferred receptacles for object type '{obj_type_lower}' (normalized to '{obj_type_lower}')")
    for key, value in SEMANTIC_RECEPTACLE_RULES.items():
        if key.lower() == obj_type_lower:
            return value
    return []


def build_semantic_map(controller) -> Dict[str, List[str]]:
    objs = controller.last_event.metadata.get("objects", []) or []
    by_id = {o.get("objectId"): o for o in objs if o.get("objectId")}

    out: Dict[str, set[str]] = {}

    for obj in objs:
        obj_type = obj.get("objectType")
        parents = obj.get("parentReceptacles") or []
        if not obj_type or not parents:
            continue

        obj_type_lower = obj_type.lower()
        out.setdefault(obj_type_lower, set())

        for pid in parents:
            parent = by_id.get(pid)
            if not parent:
                continue
            ptype = parent.get("objectType")
            if ptype:
                out[obj_type_lower].add(ptype)

    return {k: sorted(v) for k, v in out.items()}
