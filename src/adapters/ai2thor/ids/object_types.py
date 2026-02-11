
from enum import StrEnum
from typing import FrozenSet
import re

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
    Wall = "Wall"
    Ceiling = "Ceiling"

    # propperty values
    RoomTemp = "RoomTemp"
    Cold = "Cold"
    Hot = "Hot"
    Metal = "Metal"
    Wood = "Wood"
    Plastic = "Plastic"
    Glass = "Glass"
    Ceramic = "Ceramic"
    Stone = "Stone"
    Fabric = "Fabric"
    Rubber = "Rubber"
    Food = "Food"
    Paper = "Paper"
    Wax = "Wax"
    Soap = "Soap"
    Sponge = "Sponge"
    Organic = "Organic",
    True_ = "True",
    False_ = "False",


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
        ObjectType.Floor,
    ]
)

HANGING_ROLES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.ToiletPaperHanger,
        ObjectType.TowelHolder,
        ObjectType.ShowerCurtain,
        ObjectType.Wall
    ]
)

PROPERTIES_VALUES: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.RoomTemp,
        ObjectType.Cold,
        ObjectType.Hot,
        ObjectType.Metal,
        ObjectType.Wood,
        ObjectType.Plastic,
        ObjectType.Glass,
        ObjectType.Ceramic,
        ObjectType.Stone,
        ObjectType.Fabric,
        ObjectType.Rubber,
        ObjectType.Food,
        ObjectType.Paper,
        ObjectType.Wax,
        ObjectType.Soap,
        ObjectType.Sponge,
        ObjectType.Organic,
    ]
)

PICKABLE_OBJECTS: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.Apple,
        ObjectType.AppleSliced,
        ObjectType.AluminumFoil,
        ObjectType.AlarmClock,
        ObjectType.BaseballBat,
        ObjectType.BasketBall,
        ObjectType.Book,
        ObjectType.Boots,
        ObjectType.Bottle,
        ObjectType.Bowl,
        ObjectType.Box,
        ObjectType.Bread,
        ObjectType.BreadSliced,
        ObjectType.ButterKnife,
        ObjectType.CD,
        ObjectType.Candle,
        ObjectType.CellPhone,
        ObjectType.Cloth,
        ObjectType.Cup,
        ObjectType.CreditCard,
        ObjectType.DishSponge,
        ObjectType.Dumbbell,
        ObjectType.Egg,
        ObjectType.EggCracked,
        ObjectType.Fork,
        ObjectType.HandTowel,
        ObjectType.KeyChain,
        ObjectType.Knife,
        ObjectType.Kettle,
        ObjectType.Ladle,
        ObjectType.Laptop,
        ObjectType.Lettuce,
        ObjectType.LettuceSliced,
        ObjectType.Mug,
        ObjectType.Newspaper,
        ObjectType.Pen,
        ObjectType.Pencil,
        ObjectType.Pan,
        ObjectType.PaperTowelRoll,
        ObjectType.PepperShaker,
        ObjectType.Pillow,
        ObjectType.Plunger,
        ObjectType.Pot,
        ObjectType.Potato,
        ObjectType.PotatoSliced,
        ObjectType.RemoteControl,
        ObjectType.ScrubBrush,
        ObjectType.SaltShaker,
        ObjectType.SoapBar,
        ObjectType.SoapBottle,
        ObjectType.Spatula,
        ObjectType.Spoon,
        ObjectType.SprayBottle,
        ObjectType.Statue,
        ObjectType.TeddyBear,
        ObjectType.TableTopDecor,
        ObjectType.TennisRacket,
        ObjectType.TissueBox,
        ObjectType.ToiletPaper,
        ObjectType.Tomato,
        ObjectType.TomatoSliced,
        ObjectType.Vase,
        ObjectType.Towel,
        ObjectType.WateringCan,
        ObjectType.WineBottle,
        ObjectType.Watch,
    ]
)

COUNTABLE_OBJECTS: FrozenSet[ObjectType] = frozenset(
    [
        ObjectType.Apple,
        ObjectType.AppleSliced,
        ObjectType.BaseballBat,
        ObjectType.BasketBall,
        ObjectType.Book,
        ObjectType.Boots,
        ObjectType.Bottle,
        ObjectType.Bowl,
        ObjectType.Box,
        ObjectType.Bread,
        ObjectType.BreadSliced,
        ObjectType.ButterKnife,
        ObjectType.CD,
        ObjectType.CellPhone,
        ObjectType.Chair,
        ObjectType.Cloth,
        ObjectType.Cup,
        ObjectType.Curtains,
        ObjectType.DeskLamp,
        ObjectType.Desktop,
        ObjectType.DishSponge,
        ObjectType.DogBed,
        ObjectType.Dumbbell,
        ObjectType.Egg,
        ObjectType.EggCracked,
        ObjectType.Faucet,
        ObjectType.FloorLamp,
        ObjectType.Footstool,
        ObjectType.Fork,
        ObjectType.HandTowel,
        ObjectType.HandTowelHolder,
        ObjectType.HousePlant,
        ObjectType.Kettle,
        ObjectType.KeyChain,
        ObjectType.Knife,
        ObjectType.Ladle,
        ObjectType.Laptop,
        ObjectType.LaundryHamper,
        ObjectType.Lettuce,
        ObjectType.LettuceSliced,
        ObjectType.LightSwitch,
        ObjectType.Mug,
        ObjectType.Newspaper,
        ObjectType.Ottoman,
        ObjectType.Pen,
        ObjectType.Pencil,
        ObjectType.PepperShaker,
        ObjectType.Pillow,
        ObjectType.Plate,
        ObjectType.Plunger,
        ObjectType.Poster,
        ObjectType.Pot,
        ObjectType.Potato,
        ObjectType.PotatoSliced,
        ObjectType.RemoteControl,
        ObjectType.RoomDecor,
        ObjectType.SaltShaker,
        ObjectType.ScrubBrush,
        ObjectType.ShowerHead,
        ObjectType.Sofa,
        ObjectType.Spatula,
        ObjectType.Spoon,
        ObjectType.SprayBottle,
        ObjectType.Statue,
        ObjectType.Stool,
        ObjectType.TableTopDecor,
        ObjectType.TargetCircle,
        ObjectType.TeddyBear,
        ObjectType.Television,
        ObjectType.TennisRacket,
        ObjectType.TissueBox,
        ObjectType.Toaster,
        ObjectType.ToiletPaper,
        ObjectType.WineBottle,
    ]
)

ACRONYMS: FrozenSet[ObjectType] = frozenset(
    [
        "TV",
        "CD",
        "DVD",
        "USB",
        "PC",
        "CPU",
        "GPU",
        "RAM",
        "LED",
        "LCD",
        "HDMI",
        "WiFi",
        "AC",
        "DC",
    ]
)


