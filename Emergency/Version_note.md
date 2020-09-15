# Version Note

#### V11
    - I changed the weight of "pressure-reward, pres_r" due to stedy of the pressure.
      0.1 -> 1
    - I added "S/G pressure reward, SGpres_r". In experiment of V10, the agent increase 
      the posistion of "Steam Dump Valve". To restrain these actions, the agent will obtain
      the positive reward when the average of S/G pressures approach goal S/G pressure that is
      9 kg/cm^2, which is reffed in operating procedure.