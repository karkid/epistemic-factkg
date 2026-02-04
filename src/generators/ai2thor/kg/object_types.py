from enum import StrEnum
from typing import FrozenSet


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


class MaterialType(StrEnum):
    Wood = "Wood"
    Metal = "Metal"
    Plastic = "Plastic"
    Glass = "Glass"
    Fabric = "Fabric"
    Paper = "Paper"
    Rubber = "Rubber"
    Ceramic = "Ceramic"
    Stone = "Stone"
    Food = "Food"
    Organic = "Organic"
    Soap = "Soap"
    Sponge = "Sponge"
    Wax = "Wax"
    Unknown = "Unknown"


AI2THOR_OBJECT_TYPES: FrozenSet[ObjectType] = frozenset(ObjectType)

AI2THOR_CONTAINER_OBJECT_TYPES: FrozenSet[ObjectType] = frozenset(
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


AI2THOR_SURFACE_OBJECT_TYPES: FrozenSet[ObjectType] = frozenset(
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

AI2THOR_HANGING_OBJECT_TYPES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.TowelHolder,
        ObjectType.HandTowelHolder,
        ObjectType.ToiletPaperHanger,
    ]
)

AI2THOR_MATERIAL_TYPES: FrozenSet[MaterialType] = frozenset(MaterialType)
