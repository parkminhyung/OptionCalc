# Option Calculator and Strategy
<br>

Overview
--
This shiny dashboard is a simple option calculator and strategy. 

- **Option DATA :** The option data is obtained using the **yfinance** package in Python via **reticulate** package
- **Plot Package** : The plot is created using the **Plotly** package.
- This shinydashboard is a **beta version**. In the future, we plan to improve the calculation of options and greeks and add strategies. We also plan to add an option chain table

Run APPs via Rstudio or other IDEs
--
```
#install.packages(c('shiny','pacman'))
library(shiny)
runGitHub("OptionCalc","parkminhyung",ref="main")
```

Guide
--

1. When you run the Shiny app, a blank field will appear for entering a Ticker. Please enter the Ticker of a U.S. stock in this field and then click the "Fetch Data" button. <br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/0854c092-eb72-4f10-82f1-3c1bb953b7ba"> <br><br>

2. When you enter a Ticker, the blank fields in the parameters section, except for Volatility, will be automatically filled.<br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/963249a2-e555-4d87-8b33-13cc562c00f2"> <br><br>

3. The Volatility field is left blank. Once you enter the volatility of the stock, the theoretical option price will be calculated automatically on the right. This theoretical price is calculated based on the Black-Scholes model using the parameters provided. When these variables are adjusted, the theoretical option price is recalculated automatically.<br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/66af666a-9cbe-499e-abc6-d52549151038"> <br><br>


4. When you select a desired strategy in the Strategy field, the Strike and Option Price fields will expand or contract according to the chosen strategy. For strategies involving multiple strike prices, the strike prices follow the order \( k1 < k2 < k3 < k4 \). Option prices will then be calculated automatically based on these strike prices. <br>
- If you want to implement a <b>Long Call Spread</b> strategy, select <b>"LONG"</b> for the Side, <b>"CALL"</b> for the Option Type, and <b>"Spread"</b> for the Strategy. The same applies to other strategies as well.<br><br>

5. The “-” or “+” sign in front of the option prices indicates selling or buying, respectively. For example, as shown in the image below, the Jade Lizard strategy involves selling a put at strike price \( k1 \), selling a call at strike price \( k2 \), and buying a call at strike price \( k3 \), so the appropriate “-” or “+” signs are added accordingly.<br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/a2a4bd0d-d7ef-4c18-9a58-16977c60f251"> <br><br>

6. Once all parameters are filled in, click "Show Plot" in the option strategy plot section. This will display a plot and relevant details about the selected strategy, including an explanation of the strategy and its Greeks. The plot is created using the "Plotly" package. Additionally, below the plot, the strategy's profit, loss, and break-even points (BEP) are automatically calculated.<I> IV is not currently available</I><br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/01ad4222-10cc-4820-b95f-416a7295988a"> <br><br>


7. With the Plotly package, you can zoom in on the plot by dragging. To reset the plot to its original view, simply double-click on it. <br><br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/03c6d182-8f5b-438a-9291-666e24078bdd">

Available Strategies
--
- [Long / Short] Call
- [Long / Short] Put
- Covered [Call / Put]
- Protective [Call / Put]
- [Long / Short] Call Spread
- [Long / Short] Put Spread
- [Long / Short] Straddle
- [Long / Short] Strangle
- Strip
- Strap
- [Long / Short] Call Butterfly
- [Long / Short] Put Butterfly
- [Long / Short] Call Ladder
- [Long / Short] Put Ladder
- Lade Lizard / Reverse Jade Lizard
- [Long / Short] Call Condor
- [Long / Short] Put Condor

<br><br>

Contact
--
If you have any questions or suggestions, please contact : pmh621@naver.com



