import numpy as np

class Config:
    ## Exogeneous parameters generation
    mean_spot_price = 500.
    scale_spot_price = 1. 
    std_spot_price = 50.
    mean_holding_cost = 500.
    scale_holding_cost =0.15 
    std_holding_cost = 50.
    mean_waste_disposal_cost = 500.
    scale_waste_disposal_cost = 0.05 
    std_waste_disposal_cost = 50.
    mean_produced_qty = 200.
    scale_produced_qty = 1. 
    std_produced_qty = 20.
    mean_waste_qty = 200.
    scale_waste_qty = 1. 
    std_waste_qty = 20.
    buyer_init_inventory = 400.
    seller_init_inventory = 500.
    ## Coefficients for utility and cost functions

