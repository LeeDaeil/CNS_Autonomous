# Version Note

#### V12
    - I changed the weight of "pressure-reward, pres_r" due to stedy of the pressure.
      0.1 -> 1
    - I added "S/G pressure reward, SGpres_r". In experiment of V10, the agent increase 
      the posistion of "Steam Dump Valve". To restrain these actions, the agent will obtain
      the positive reward when the average of S/G pressures approach goal S/G pressure that is
      9 kg/cm^2, which is reffed in operating procedure.
#### V13
    Topic: The S/G pressuire is not reduced.

    - The agent cannot control the press setpoint to reach the maximum S/G pressure. As a reason,
      If then logic cannot drop the press setpoint. Therefore, I changed the control position per time.
            + I added two key value, (IFLOGIC_SteamDumpUp and IFLOGIC_SteamDumpDown), in V.
            + I not olny added SteamDumpRate but also modified that value as 4. (2 -> 4) 
      
    - The agent shows the minimum change of Aux flow, when the S/G level is high. I need to consider
      two ways;
        * Does it need the S/G wide range? The agent uses the narrow range parameters, which means
          that the agent gets a zero value until the S/G narrow range is restored.
            + -- remove this idea --
        * Does agent's control rate require to increase? The chage of Aux feed water flow is 0.2 per
          one click. It cannot expect to the good resonponce to calculating the reward.       
            + 0.2 -> 0.4, I added "RL_IncreaseAux(1,2,3)Flow".
#### V14
    Topic: Trajection of PZR pressure stray from PT curve.
    
    - In V13, the agent followed cooling rate. Problem is the trajection straied from PT curve.
      * Does pressure reward's of weight require to decrease?
      * How can the agent know that they have to follow the reward?
      * Should I add more agent to achive the goal? 
    
    Topic: Using positive reward give good performance better than using negative reward.
           [ref: https://stackoverflow.com/questions/49801638/normalizing-rewards-
                 to-generate-returns-in-reinforcement-learning]
    - ...
    - ...
#### V_ex ...
    Topic: ...
    
    - ...
    - ... 