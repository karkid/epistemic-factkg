from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet

from src.core.registry.entity import EntityInfo, EntityRegistry, SpatialRole


class ObjectType(StrEnum):
    AlarmClock = "AlarmClock"
    AluminumFoil = "AluminumFoil"
    Apple = "Apple"
    AppleSliced = "AppleSliced"
    ArmChair = "ArmChair"
    BaseballBat = "BaseballBat"
    BasketBall = "BasketBall"
    Bathtub = "Bathtub"
    BathtubBasin = "BathtubBasin"
    Bed = "Bed"
    Blinds = "Blinds"
    Book = "Book"
    Boots = "Boots"
    Bottle = "Bottle"
    Bowl = "Bowl"
    Box = "Box"
    Bread = "Bread"
    BreadSliced = "BreadSliced"
    ButterKnife = "ButterKnife"
    Cabinet = "Cabinet"
    Candle = "Candle"
    CD = "CD"
    CellPhone = "CellPhone"
    Chair = "Chair"
    Cloth = "Cloth"
    CoffeeMachine = "CoffeeMachine"
    CoffeeTable = "CoffeeTable"
    CounterTop = "CounterTop"
    CreditCard = "CreditCard"
    Cup = "Cup"
    Curtains = "Curtains"
    Desk = "Desk"
    DeskLamp = "DeskLamp"
    Desktop = "Desktop"
    DiningTable = "DiningTable"
    DishSponge = "DishSponge"
    DogBed = "DogBed"
    Drawer = "Drawer"
    Dresser = "Dresser"
    Dumbbell = "Dumbbell"
    Egg = "Egg"
    EggCracked = "EggCracked"
    Faucet = "Faucet"
    Floor = "Floor"
    FloorLamp = "FloorLamp"
    Footstool = "Footstool"
    Fork = "Fork"
    Fridge = "Fridge"
    GarbageBag = "GarbageBag"
    GarbageCan = "GarbageCan"
    HandTowel = "HandTowel"
    HandTowelHolder = "HandTowelHolder"
    HousePlant = "HousePlant"
    Kettle = "Kettle"
    KeyChain = "KeyChain"
    Knife = "Knife"
    Ladle = "Ladle"
    Laptop = "Laptop"
    LaundryHamper = "LaundryHamper"
    Lettuce = "Lettuce"
    LettuceSliced = "LettuceSliced"
    LightSwitch = "LightSwitch"
    Microwave = "Microwave"
    Mirror = "Mirror"
    Mug = "Mug"
    Newspaper = "Newspaper"
    Ottoman = "Ottoman"
    Painting = "Painting"
    Pan = "Pan"
    PaperTowelRoll = "PaperTowelRoll"
    Pen = "Pen"
    Pencil = "Pencil"
    PepperShaker = "PepperShaker"
    Pillow = "Pillow"
    Plate = "Plate"
    Plunger = "Plunger"
    Poster = "Poster"
    Pot = "Pot"
    Potato = "Potato"
    PotatoSliced = "PotatoSliced"
    RemoteControl = "RemoteControl"
    RoomDecor = "RoomDecor"
    Safe = "Safe"
    SaltShaker = "SaltShaker"
    ScrubBrush = "ScrubBrush"
    Shelf = "Shelf"
    ShelvingUnit = "ShelvingUnit"
    ShowerCurtain = "ShowerCurtain"
    ShowerDoor = "ShowerDoor"
    ShowerGlass = "ShowerGlass"
    ShowerHead = "ShowerHead"
    SideTable = "SideTable"
    Sink = "Sink"
    SinkBasin = "SinkBasin"
    SoapBar = "SoapBar"
    SoapBottle = "SoapBottle"
    Sofa = "Sofa"
    Spatula = "Spatula"
    Spoon = "Spoon"
    SprayBottle = "SprayBottle"
    Statue = "Statue"
    Stool = "Stool"
    StoveBurner = "StoveBurner"
    StoveKnob = "StoveKnob"
    TableTopDecor = "TableTopDecor"
    TargetCircle = "TargetCircle"
    TeddyBear = "TeddyBear"
    Television = "Television"
    TennisRacket = "TennisRacket"
    TissueBox = "TissueBox"
    Toaster = "Toaster"
    Toilet = "Toilet"
    ToiletPaper = "ToiletPaper"
    ToiletPaperHanger = "ToiletPaperHanger"
    Tomato = "Tomato"
    TomatoSliced = "TomatoSliced"
    Towel = "Towel"
    TowelHolder = "TowelHolder"
    TVStand = "TVStand"
    VacuumCleaner = "VacuumCleaner"
    Vase = "Vase"
    Watch = "Watch"
    WateringCan = "WateringCan"
    Window = "Window"
    WineBottle = "WineBottle"


CONTAINER_ROLES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.Fridge,
        ObjectType.Microwave,
        ObjectType.Cabinet,
        ObjectType.Drawer,
        ObjectType.Safe,
        ObjectType.Toilet,
        ObjectType.Box,
        ObjectType.SinkBasin,
        ObjectType.BathtubBasin,
    ]
)

SURFACE_ROLES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.StoveBurner,
        ObjectType.CounterTop,
        ObjectType.DiningTable,
        ObjectType.CoffeeTable,
        ObjectType.SideTable,
        ObjectType.Desk,
        ObjectType.Dresser,
        ObjectType.TVStand,
        ObjectType.Shelf,
        ObjectType.Sofa,
        ObjectType.ArmChair,
        ObjectType.Ottoman,
        ObjectType.Bed,
    ]
)

HANGING_ROLES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.ToiletPaperHanger
    ]
)


def create_entity_registry(registry: EntityRegistry) -> None:
    """
    Populate a core EntityRegistry with AI2-THOR object vocabulary.
    """
    for obj in ObjectType:
        if obj in CONTAINER_ROLES:
            role = SpatialRole.CONTAINER
        elif obj in SURFACE_ROLES:
            role = SpatialRole.SURFACE
        elif obj in HANGING_ROLES:
            role = SpatialRole.HANGING
        else:
            role = None

        registry.register(
            obj.value,
            EntityInfo(
                name=obj.value,
                type="object",
                is_countable=True,
                spatial_roles=role,
            ),
        )
