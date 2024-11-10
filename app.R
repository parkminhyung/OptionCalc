

required_packages <- c(
  "shiny",
  "shinydashboard",
  "shinythemes",
  "reticulate",
  "dplyr",
  "tibble",
  "shinycssloaders",
  "ggplot2",
  "pacman",
  "plotly",
  "purrr"
)


pacman::p_load(char = required_packages, install = TRUE)


ui <- dashboardPage(
  dashboardHeader(title = "Option Calculator"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Option Calculator & Strategy", tabName = "option_calculator", icon = icon("calculator")),
      menuItem("About", tabName = "about", icon = icon("circle-info"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$style(HTML("
        table {
          width: 100%;
          table-layout: auto;
        }
        th, td {
          padding: 12px;
          text-align: center;
          font-size: 16px;
        }
        .large-font {
          font-size: 24px;
          font-weight: bold;
        }
        .small-font {
          font-size: 16px;
        }
        .bold-left-align {
          font-weight: bold;
          text-align: left;
        }
      "))
    ),
    tabItems(
      tabItem(tabName = "option_calculator",
              fluidRow(
                column(12,
                       box(title = "Information", width = 12, status = "primary", solidHeader = TRUE,
                           textInput("ticker", "TICKER:", value = NULL),
                           actionButton("fetch", "Fetch Data"),
                           br(),
                           br(),
                           uiOutput("stock_name"),  
                           uiOutput("sector"),      
                           br(),
                           fluidRow(
                             column(1, htmlOutput("open")),
                             column(1, htmlOutput("high")),
                             column(1, htmlOutput("low")),
                             column(1, htmlOutput("close")),
                             column(1, htmlOutput("chg")),
                             column(1, htmlOutput("vol"))
                           )
                       )
                )
              ),
              fluidRow(
                column(12,
                       box(
                         title = "Parameters",
                         width = 12,
                         status = "primary",
                         solidHeader = TRUE,
                         fluidRow(
                           column(3,
                                  numericInput("s", "Underlying Price (S)", value = NULL, step = 0.01),
                                  selectInput("expiry", "Expiry Date", choices = NULL),
                                  numericInput("tau", "DTE (days)", value = NULL),
                                  numericInput("rf", "Risk-free Rate (%)", value = NULL),
                                  numericInput("y", "Dividend Yield (%)", value = NULL)
                           ),
                           column(3, 
                                  selectInput("side", "Side", choices = c("LONG","SHORT")),
                                  selectInput("option_type", "Option Type", choices = c("CALL","PUT")),
                                  selectInput("strat", "Strategy", choices = c("Single", "Covered" , "Protective" , "Spread" , "Straddle", "Strangle","Strip","Strap","Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard","Condor")),
                                  selectInput("greeks", "Greeks", choices = c("Delta","Gamma","Vega","Theta","Rho")),
                                  numericInput("sigma", "Volatility (%)", value = NULL)
                                  
                           ),
                           column(3, 
                                  uiOutput("strike_ui"),
                                  numericInput("size", "Size (@100)", value = NULL)
                           ),
                           column(3, 
                                  uiOutput("price_ui")  
                           )
                         )
                       )
                )
              ),
              fluidRow(
                column(12,
                       box(
                         title = "Option Strategy Plot", 
                         width = 12, 
                         status = "primary", 
                         solidHeader = TRUE,
                         actionButton(inputId = "plot_button", label = "Show Plot"),
                         br(),
                         htmlOutput("greeks_table"),
                         br(),
                         withSpinner(plotlyOutput("strategy_plot", height = "400px")),  
                         br(),
                         htmlOutput("bep_value")
                       )
                )
              )
              
      ),
      
      tabItem(tabName = "about",  
              fluidRow(
                column(12,
                       box(title = "About Pricing Models", width = 12, status = "primary", solidHeader = TRUE,
                           withMathJax(),
                           HTML("
                 <h3><b>Option pricing Model: Black-Scholes Model </b></h3>
          The theoretical option price was calculated using the <b>Black-Scholes model</b>.  
          The Black-Scholes model is the most widely used option pricing model and was developed by Fisher Black and Myron Scholes to derive European option prices based on Einstein's Brownian motion equations.  
          The formula used is shown below: <br><br>

          $$
          \\begin{aligned}
          d_1 &= \\frac{ln(S_0/K) + (r_f - y + 0.5 \\sigma^2) \\tau}{\\sigma \\sqrt{\\tau}} 
          \\\\
          d_2 &= d_1 - \\sigma \\sqrt{\\tau}
          \\end{aligned}
          $$
          
          <b> Call price: </b>

          $$
          C(S_0, \\tau) = S_0 N(d_1) e^{-y \\tau} - K e^{-r_f \\tau} N(d_2)
          $$

          <b> Put price: </b>

          $$
          P(S_0, \\tau) = K e^{-r_f \\tau} N(-d_2) - S_0 N(-d_1) e^{-y \\tau}
          $$

          <p> <i>where, </i></p>  
            - S<sub>0</sub> : Underlying Price <br>
            - K : Strike Price <br>
            - r<sub>f</sub> : Risk- free rate <br>
            - y : Dividend yield <br>
            - τ : Time to maturity <br>
            - N(x) : Standard normal cumulative distribution function <br>

          <br>
          <br>

          <h3><b>Option pricing Model: The Black model (Black 76' model) </b></h3>
          
          The Black model (or Black-76 model) was originally developed as an option pricing formula for non-traded assets, such as commodity options. 
          By using the forward price as the underlying asset instead of the stock price, the need for a hedging process was eliminated, making it useful for pricing options on assets that are not traded in the market. However, it has become more widely used for valuing futures options, bond options, and interest rate options, which are in higher demand.
          The Black model essentially uses the futures price (F<sub>0</sub>) instead of the spot price (S<sub>0</sub>).
          The black model equation is shown below: <br><br>

          $$
          \\begin{aligned}
          d_1 &= \\frac{ln(F/K)+(\\sigma_{F}^2/2)\\tau}{\\sigma_{F} \\sqrt{\\tau}}
          \\\\
          d_2 &= \\sigma_{F} \\sqrt{\\tau}
          \\end{aligned}
          $$

          <b> Call price: </b>
          \\begin{aligned}
          C(F,\\tau) &= e^{-r_f\\tau}[FN(d_1) - KN(d_2)] 
          \\end{aligned}


            <b> Put price: </b>
            \\begin{aligned}
            P(F,\\tau) &= e^{-r_f\\tau}[KN(-d_2)-FN(-d_1)]
            \\end{aligned}


            <p> <i>where, </i></p>  <br>
            - σ<sub>F</sub> :  future's sigma <br>
            - F(Future price) = S<sub>0</sub><sup>r<sub>f</sub>τ</sup>

            <br>
            <br>
            
            
            <h3><b>Option Greeks </b></h3>
            Option Greeks are key measures that assess an option's price sensitivity to factors like volatility and the price of the underlying asset. They are crucial for analyzing options portfolios and are widely used by investors to make informed trading decisions.
            <br>
            <br>
            <b>Delta:</b><br>
            Delta measures how much an option's price will change for every $1 movement in the underlying asset. A Delta of 0.40 means the option price will move $0.40 for each $1 change, and suggests a 40% chance the option will expire in the money (ITM).<br>
            Call options have a Delta between 0.00 and 1.00, with at-the-money options near 0.50, increasing toward 1.00 as they move deeper ITM or approach expiration, and decreasing toward 0.00 if they are out-of-the-money. <br>
            Put options have a Delta between 0.00 and –1.00, with at-the-money options near –0.50, decreasing toward –1.00 as they move deeper ITM or approach expiration, and approaching 0.00 if they are out-of-the-money.<br>
            <br>
            
            
            \\begin{aligned}
            Call:
            \\Delta_c &= e^{-y\\tau}N(d_1) 
            \\\\
            \\\\
            Put:
            \\Delta_p &= e^{-y\\tau}[N(d_1)-1] 
            \\end{aligned}

            \

            
            <b>Gamma:</b><br>
            Gamma measures the rate of change in an option's Delta for every $1 move in the underlying asset, much like acceleration compared to speed. As Delta increases with a stock price move, Gamma reflects how much Delta shifts. For example, if Delta changes from 0.40 to 0.55 after a $1 move, the Gamma is 0.15. As options get deeper ITM and Delta approaches 1.00, Gamma decreases since there's less room for further acceleration.<br>
            <br>
            
            
            \\begin{aligned}
            \\Gamma &= \\frac{e^{-y\\tau}}{S_0\\sigma\\sqrt{\\tau}}N'(d_1)
            \\end{aligned}

            \

            
            <b>Vega:</b><br>
            Vega measures the change in an option's price for a one-percentage-point change in the implied volatility of the underlying asset. It indicates how much an option's price is expected to move with changes in volatility, which is crucial for option valuation. Typically, a decrease in Vega results in a loss of value for both calls and puts, while an increase in Vega leads to a gain in value for both types of options.<br>
            <br>
            
            \\begin{aligned}
            \\nu &= S_0N'(d_1)\\sqrt{\\tau}
            \\end{aligned}

            \

            
            <b>Theta:</b><br>
            Theta measures the daily decrease in an option's price as it approaches expiration, reflecting time decay. This erosion is not linear; the price decline accelerates for at-the-money (ATM) and slightly out-of-the-money options as expiration nears, while the decline in far out-of-the-money options typically slows down.<br>
            <br>
            
            \\begin{aligned}
            Call : 
            \\theta_c &= 1/T(-(\\frac{S_0\\sigma e^{-y\\tau}}{2\\sqrt{\\tau}}N'(d_1)) - r_fKe^{r_f\\tau} N(d_2) + yS_0e^{-y\\tau}N(d_1)) 
            \\\\
            \\\\
            Put : 
            \\theta_p &= 1/T(-(\\frac{S_0\\sigma e^{-y\\tau}}{2\\sqrt{\\tau}}N'(d_1)) + r_fKe^{r_f\\tau} N(-d_2) - yS_0e^{-y\\tau}N(-d_1))
            \\end{aligned}

            \

            <b>Rho:</b><br>
            Rho measures the change in an option's price for a one-percentage-point shift in interest rates, indicating how the price will adjust with changes in the risk-free interest rate. Generally, as interest rates rise, call options increase in value, leading to positive Rho, while put options typically decrease in value, resulting in negative Rho.
            <br><br>
            
            \\begin{aligned}
            Call: \\rho_c &= K\\tau e^{-r_f\\tau}N(d_2) 
            \\\\
            \\\\
            Put: \\rho_p &= -K\\tau e^{-r_f\\tau}N(-d_2) 
            \\end{aligned}

            <br>
            <br>

            <h3><b> Implied Volatility </b></h3>
            Implied volatility refers to the volatility of the underlying asset (such as a stock) that is embedded in the option price. Implied volatility, derived from multiple options, is often used as an estimate of future market volatility. Typically, implied volatility increases in a bearish market, when most investors believe that the asset’s price will decline over a certain period. On the other hand, when market expectations are bullish, implied volatility tends to decrease. This is based on the widely held belief that bear markets are riskier than bull markets.<br>
            <br>
            Implied volatility is derived from the Black-Scholes model by inserting the market price of the option into the theoretical pricing model and solving for volatility in reverse. It can be interpreted as the market’s estimate of the asset’s volatility until the option’s expiration. Since there is no closed-form solution for calculating implied volatility, it must be obtained through numerical methods, such as the Newton-Raphson method.
            <br>
            <br>

             $$
            \\begin{aligned}
            BSM : f(S_0,K,r_f,y,\\tau,\\sigma) &= C_{theor. price}
            \\\\
            \\\\
            IV : f^{-1}(C_{market}) &= \\sigma_{IV}  
            \\end{aligned}
            $$

            <br>
            <br>
            <br>
            <span style = 'font-size:90%'>If you have any questions or suggestions, please contact : pmh621@naver.com </span>

                
              ")
                       )
                )
              )
      )
    )
  ),
  skin = "purple"
)




# Setting pyton path
options(scipen = 999)

py_run_string("
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.info

    underlying_price = stock_info.get('currentPrice', None)

    rf = round(yf.Ticker('^TNX').history(period='1d').iloc[0,3],3) if 'currentPrice' in stock_info else 3.95
    dividend_yield = stock_info.get('dividendYield', 0.0) * 100

    open_price = stock_info.get('open', None)
    high = stock_info.get('dayHigh', None)
    low = stock_info.get('dayLow', None)

    previous_close = stock_info.get('previousClose', None)
    chg = round(((underlying_price / previous_close) - 1) * 100, 3) if previous_close else 0.0

    name = stock_info.get('shortName', 'N/A')
    sector = stock_info.get('sector', 'N/A')

    option_chain = stock.option_chain()
    expiry_date = stock.options 

    return underlying_price, rf, dividend_yield, option_chain.calls, option_chain.puts, expiry_date, open_price, high, low, chg, name, sector

def get_52weeks_volatility(ticker):

    stock_data = yf.download(ticker, period='1y')
    stock_data['Returns'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    stock_data.dropna(inplace=True)
    daily_volatility = stock_data['Returns'].std()
    annual_volatility = daily_volatility * np.sqrt(252)

    return round(annual_volatility * 100, 2)

def option_chain(expiry_date, ticker):
    stock = yf.Ticker(ticker)
    call_chain = stock.option_chain(expiry_date)[0]
    put_chain = stock.option_chain(expiry_date)[1]

    return call_chain, put_chain
")


bs_model <- function(s, k, rf, tau, sigma, y, option_type = "c") {
  
  T = 252
  pct = 100
  tau <- tau/T
  sigma <- sigma/pct
  rf <- rf/pct
  y <- y/pct
  
  d1 <- (log(s / k) + (rf - y + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau))
  d2 <- d1 - sigma * sqrt(tau)
  
  call_price <- s * exp(-y * tau) * pnorm(d1) - k * exp(-rf * tau) * pnorm(d2)
  put_price <- k * exp(-rf * tau) * pnorm(-d2) - s * exp(-y * tau) * pnorm(-d1)
  
  if (option_type == "c") {
    return(call_price)
  } else if (option_type == "p") {
    return(put_price)
  } else {
    return(list(call_price = call_price, put_price = put_price))
  }
}


option_greeks <- function(s, k, rf, sigma, tau, y) {
  
  T = 252
  pct = 100
  
  tau <- tau/T
  sigma <- sigma/pct
  rf <- rf/pct
  y <- y/pct
  
  d1 <- (log(s / k) + (rf - y + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
  d2 <- d1 - (sigma * sqrt(tau))
  
  nd1 <- (1 / (sqrt(2 * pi))) * exp(-(d1^2 / 2))
  
  # Delta
  call_delta <- pnorm(d1)
  put_delta <- (pnorm(d1) - 1)
  
  # Gamma
  gamma <- (nd1) / (s * sigma * sqrt(tau))
  
  # Theta
  call_theta <- (-((s * sigma) / (2 * sqrt(tau))) * nd1 - k * exp(-rf * tau) * rf * pnorm(d2))/T
  put_theta <- (-((s * sigma) / (2 * sqrt(tau))) * nd1 - k * exp(-rf * tau) * rf * (pnorm(d2) - 1))/T
  
  # Rho
  call_rho <- (k * tau * exp(-rf * tau) * pnorm(d2))/pct
  put_rho <- (k * tau * exp(-rf * tau) * (pnorm(d2) - 1))/pct
  
  # Vega
  vega <- (s * sqrt(tau) * nd1)/pct
  
  return(list(
    call.Delta = call_delta,
    put.Delta = put_delta,
    Gamma = gamma,
    Vega = vega,
    call.Theta = call_theta,
    put.Theta = put_theta,
    call.Rho = call_rho,
    put.Rho = put_rho
  ))
}

greek_df = function(chain_df,s,rf,y,sigma,k1,k2=NA,k3=NA,k4=NA,price1,price2=NA,price3=NA,price4=NA,size,tau,option_type,side,strat){
  
  chain_df = chain_df %>%
    select(contractSymbol, type, everything())
  
  df <- tibble(
    x = chain_df %>%
      filter(type == option_type) %>%
      .$strike %>%
      {
        seq(min(.), max(.), by = .1)
      }
  )
  
  p1 = ifelse(df$x <= k1, ((k1 - df$x) - price1), -price1)
  c1 = ifelse(df$x <= k1, -price1, ((df$x - k1) - price1))
  
  k1 <- ifelse(is.na(k1),0,k1)
  k2 <- ifelse(is.na(k2),0,k2)
  k3 <- ifelse(is.na(k3),0,k3)
  k4 <- ifelse(is.na(k4),0,k4)
  
  if(!is.na(price2)) {
    p2 = ifelse(df$x <= k2, ((k2 - df$x) - price2), -price2)
    c2 = ifelse(df$x <= k2, -price2, ((df$x - k2) - price2))
  } else {
    p2 = 0
    c2 = 0
  }
  
  if(!is.na(price3)) {
    p3 = ifelse(df$x <= k3, ((k3 - df$x) - price3), -price3)
    c3 = ifelse(df$x <= k3, -price3, ((df$x - k3) - price3))
  } else {
    p3 = 0
    c3 = 0
  }
  
  if(!is.na(price4)){
    p4 = ifelse(df$x <= k4, ((k4 - df$x) - price4), -price4)
    c4 = ifelse(df$x <= k4, -price4, ((df$x - k4) - price4))
  } else {
    p4 = 0
    c4 = 0
  }
  
  # Greeks
  c.d1 = option_greeks(df$x,k1,rf,sigma,tau,y)$call.Delta
  c.d2 = option_greeks(df$x,k2,rf,sigma,tau,y)$call.Delta
  c.d3 = option_greeks(df$x,k3,rf,sigma,tau,y)$call.Delta
  c.d4 = option_greeks(df$x,k4,rf,sigma,tau,y)$call.Delta
  
  p.d1 = option_greeks(df$x,k1,rf,sigma,tau,y)$put.Delta
  p.d2 = option_greeks(df$x,k2,rf,sigma,tau,y)$put.Delta
  p.d3 = option_greeks(df$x,k3,rf,sigma,tau,y)$put.Delta
  p.d4 = option_greeks(df$x,k4,rf,sigma,tau,y)$put.Delta
  
  g1 = option_greeks(df$x,k1,rf,sigma,tau,y)$Gamma
  g2 = option_greeks(df$x,k2,rf,sigma,tau,y)$Gamma
  g3 = option_greeks(df$x,k3,rf,sigma,tau,y)$Gamma
  g4 = option_greeks(df$x,k4,rf,sigma,tau,y)$Gamma
  
  v1 = option_greeks(df$x,k1,rf,sigma,tau,y)$Vega
  v2 = option_greeks(df$x,k2,rf,sigma,tau,y)$Vega
  v3 = option_greeks(df$x,k3,rf,sigma,tau,y)$Vega
  v4 = option_greeks(df$x,k4,rf,sigma,tau,y)$Vega
  
  c.th1 = option_greeks(df$x,k1,rf,sigma,tau,y)$call.Theta
  c.th2 = option_greeks(df$x,k2,rf,sigma,tau,y)$call.Theta
  c.th3 = option_greeks(df$x,k3,rf,sigma,tau,y)$call.Theta
  c.th4 = option_greeks(df$x,k4,rf,sigma,tau,y)$call.Theta
  
  p.th1 = option_greeks(df$x,k1,rf,sigma,tau,y)$put.Theta
  p.th2 = option_greeks(df$x,k2,rf,sigma,tau,y)$put.Theta
  p.th3 = option_greeks(df$x,k3,rf,sigma,tau,y)$put.Theta
  p.th4 = option_greeks(df$x,k4,rf,sigma,tau,y)$put.Theta
  
  c.rh1 = option_greeks(df$x,k1,rf,sigma,tau,y)$call.Rho
  c.rh2 = option_greeks(df$x,k2,rf,sigma,tau,y)$call.Rho
  c.rh3 = option_greeks(df$x,k3,rf,sigma,tau,y)$call.Rho
  c.rh4 = option_greeks(df$x,k4,rf,sigma,tau,y)$call.Rho
  
  p.rh1 = option_greeks(df$x,k1,rf,sigma,tau,y)$put.Rho
  p.rh2 = option_greeks(df$x,k2,rf,sigma,tau,y)$put.Rho
  p.rh3 = option_greeks(df$x,k3,rf,sigma,tau,y)$put.Rho
  p.rh4 = option_greeks(df$x,k4,rf,sigma,tau,y)$put.Rho
  
  df <- df %>%
    mutate(
      y = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ c1,
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ - c1,
        
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ p1,
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ - p1,
        
        strat == "Straddle" & side == "LONG" ~  p1 + c1,
        strat == "Straddle" & side == "SHORT" ~ - p1 - c1,
        
        
        strat == "Strangle" & side == "LONG" ~ p1 + c2,
        strat == "Strangle" & side == "SHORT" ~ - p1 - c2,
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ c1 - c2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ - c1 + c2,
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ p1 - p2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ - p1 + p2,
        
        
        
        strat == "Covered" & option_type == "PUT" ~ (s - x) - p1,
        
        
        strat == "Covered" & option_type == "CALL" ~ (x - s) - c1,
        
        
        strat == "Protective" & option_type == "PUT" ~ (x - s) + p1,
        
        
        strat == "Protective" & option_type == "CALL" ~ (s - x) + c1,
        
        strat == "Strip" ~ c1 + 2*p1,
        strat == "Strap" ~ 2*c1 + p1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ c1 - 2*c2 + c3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ -(c1 - 2*c2 + c3),
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ p1 - 2*p2 + p3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(p1 - 2*p2 + p3),
        
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ c1 - c2 -c3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(c1 - c2 -c3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ p1 + p2 - p3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(p1 + p2 - p3),
        
        strat == "Jade Lizard" ~ -p1 -c2 + c3,
        strat == "Reverse Jade Lizard" ~ +p1 -p2 -c3,
        
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ c1 - c2 - c3 + c4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(c1 - c2 - c3 + c4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  p1 - p2 - p3 + p4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(p1 - p2 - p3 + p4)
        
        
      ) %>% `*`(size) %>% round(.,digits = 5),
      
      
      bep1 = ifelse(length(x[which(y * lag(y, default = first(y)) < 0)]) >= 1, round(x[which(y * lag(y, default = first(y)) < 0)][1], digits = 5), NA),
      
      
      bep2 = ifelse(length(x[which(y * lag(y, default = first(y)) < 0)]) >= 2, round(x[which(y * lag(y, default = first(y)) < 0)][2], digits = 5), NA),
      
      Delta = case_when(
        # Single Option Strategies
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ (c.d1),
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ (-c.d1),
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ (p.d1),
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ (-p.d1),
        
        # Straddle Strategy
        strat == "Straddle" & side == "LONG" ~  c.d1 + p.d1,
        strat == "Straddle" & side == "SHORT" ~ -(c.d1 + p.d1),
        
        
        strat == "Strangle" & side == "LONG" ~ (c.d1 + p.d2),
        strat == "Strangle" & side == "SHORT" ~ -(c.d1 + p.d2),
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ c.d1 - c.d2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ -(c.d1 - c.d2),
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ p.d1 - p.d2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ -(p.d1 - p.d2),
        
        
        
        strat == "Covered" & option_type == "PUT" ~ -1 + p.d1,
        
        
        strat == "Covered" & option_type == "CALL" ~ 1 - c.d1,
        
        
        strat == "Protective" & option_type == "PUT" ~ 1 + p.d1,
        
        
        strat == "Protective" & option_type == "CALL" ~ -1 + c.d1,
        
        strat == "Strip" ~ c.d1 + 2*p.d1,
        strat == "Strap" ~ 2*c.d1 + p.d1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ c.d1 - (2*c.d2) + c.d3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ - c.d1 + (2*c.d2) - c.d3,
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ p.d1 - 2*p.d2 + p.d3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(p.d1 - 2*p.d2 + p.d3),
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ c.d1 - c.d2 -c.d3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(c.d1 - c.d2 -c.d3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ p.d1 + p.d2 - p.d3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(p.d1 + p.d2 - p.d3),
        
        strat == "Jade Lizard" ~ -p.d1 -c.d2 + c.d3,
        strat == "Reverse Jade Lizard" ~ p.d1 -p.d2 -c.d3,
        
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ c.d1 - c.d2 - c.d3 + c.d4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(c.d1 - c.d2 - c.d3 + c.d4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  p.d1 - p.d2 - p.d3 + p.d4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(p.d1 - p.d2 - p.d3 + p.d4)
        
        
      ) %>% `*`(size) %>% round(., digits = 5) ,
      
      Gamma = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ g1,
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ - g1,
        
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ g1,
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ - g1,
        
        strat == "Straddle" & side == "LONG" ~  g1 + g1,
        strat == "Straddle" & side == "SHORT" ~ - g1 - g1,
        
        
        strat == "Strangle" & side == "LONG" ~ g1 + g2,
        strat == "Strangle" & side == "SHORT" ~ - g1 - g2,
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ g1 - g2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ - g1 + g2,
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ g1 - g2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ - g1 + g2,
        
        
        
        strat == "Covered" & option_type == "PUT" ~ g1,
        
        
        strat == "Covered" & option_type == "CALL" ~ - g1,
        
        
        strat == "Protective" & option_type == "PUT" ~ g1,
        
        
        strat == "Protective" & option_type == "CALL" ~ g1,
        
        strat == "Strip" ~ 3*g1,
        strat == "Strap" ~ 3*g1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ g1 - 2*g2 + g3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ -(g1 - 2*g2 + g3),
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ g1 - 2*g2 + g3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(g1 - 2*g2 + g3),
        
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ g1 - g2 -g3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(g1 - g2 -g3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ g1 + g2 - g3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(g1 + g2 - g3),
        
        strat == "Jade Lizard" ~ -g1 -g2 + g3,
        strat == "Reverse Jade Lizard" ~ g1 -g2 -g3,
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ g1 - g2 - g3 + g4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(g1 - g2 - g3 + g4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  g1 - g2 - g3 + g4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(g1 - g2 - g3 + g4)
        
        
        
      ) %>% `*`(size) %>% round(., digits = 5),
      
      Vega = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ v1,
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ - v1,
        
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ v1,
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ - v1,
        
        strat == "Straddle" & side == "LONG" ~  v1 + v1,
        strat == "Straddle" & side == "SHORT" ~ - v1 - v1,
        
        
        strat == "Strangle" & side == "LONG" ~ v1 + v2,
        strat == "Strangle" & side == "SHORT" ~ - v1 - v2,
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ v1 - v2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ - v1 + v2,
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ v1 - v2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ - v1 + v2,
        
        
        
        strat == "Covered" & option_type == "PUT" ~ v1,
        
        
        strat == "Covered" & option_type == "CALL" ~ - v1,
        
        
        strat == "Protective" & option_type == "PUT" ~ v1,
        
        
        strat == "Protective" & option_type == "CALL" ~ v1,
        
        strat == "Strip" ~ 3*v1,
        strat == "Strap" ~ 3*v1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ v1 - 2*v2 + v3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ -(v1 - 2*v2 + v3),
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ v1 - 2*v2 + v3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(v1 - 2*v2 + v3),
        
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ v1 - v2 -v3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(v1 - v2 -v3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ v1 + v2 - v3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(v1 + v2 - v3),
        
        strat == "Jade Lizard" ~ -v1 -v2 + v3,
        strat == "Reverse Jade Lizard" ~ v1 -v2 -v3,
        
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ v1 - v2 - v3 + v4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(v1 - v2 - v3 + v4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  v1 - v2 - v3 + v4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(v1 - v2 - v3 + v4)
        
        
      ) %>% `*`(size) %>% round(., digits = 5),
      
      Theta = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ c.th1,
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ - c.th1,
        
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ p.th1,
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ - p.th1,
        
        strat == "Straddle" & side == "LONG" ~  p.th1 + c.th1,
        strat == "Straddle" & side == "SHORT" ~ - p.th1 - c.th1,
        
        
        strat == "Strangle" & side == "LONG" ~ p.th1 + c.th2,
        strat == "Strangle" & side == "SHORT" ~ - p.th1 - c.th2,
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ c.th1 - c.th2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ - c.th1 + c.th2,
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ p.th1 - p.th2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ - p.th1 + p.th2,
        
        
        
        strat == "Covered" & option_type == "PUT" ~ - p.th1,
        
        
        strat == "Covered" & option_type == "CALL" ~ - c.th1,
        
        
        strat == "Protective" & option_type == "PUT" ~ p.th1,
        
        
        strat == "Protective" & option_type == "CALL" ~ c.th1,
        
        strat == "Strip" ~ c.th1 + 2*p.th1,
        strat == "Strap" ~ 2*c.th1 + p.th1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ c.th1 - 2*c.th2 + c.th3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ -(c.th1 - 2*c.th2 + c.th3),
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ p.th1 - 2*p.th2 + p.th3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(p.th1 - 2*p.th2 + p.th3),
        
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ c.th1 - c.th2 -c.th3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(c.th1 - c.th2 -c.th3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ p.th1 + p.th2 - p.th3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(p.th1 + p.th2 - p.th3),
        
        strat == "Jade Lizard" ~ -p.th1 -c.th2 + c.th3,
        strat == "Reverse Jade Lizard" ~ p.th1 -p.th2 -c.th3,
        
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ c.th1 - c.th2 - c.th3 + c.th4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(c.th1 - c.th2 - c.th3 + c.th4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  p.th1 - p.th2 - p.th3 + p.th4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(p.th1 - p.th2 - p.th3 + p.th4)
        
        
      ) %>% `*`(size) %>% round(., digits = 5),
      
      Rho = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ c.rh1,
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ - c.rh1,
        
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ p.rh1,
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ - p.rh1,
        
        strat == "Straddle" & side == "LONG" ~  p.rh1 + c.rh1,
        strat == "Straddle" & side == "SHORT" ~ - p.rh1 - c.rh1,
        
        
        strat == "Strangle" & side == "LONG" ~ p.rh1 + c.rh2,
        strat == "Strangle" & side == "SHORT" ~ - p.rh1 - c.rh2,
        
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ c.rh1 - c.rh2,
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ - c.rh1 + c.rh2,
        
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ p.rh1 - p.rh2,
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ - p.rh1 + p.rh2,
        
        
        
        strat == "Covered" & option_type == "PUT" ~ - p.rh1,
        
        
        strat == "Covered" & option_type == "CALL" ~ - c.rh1,
        
        
        strat == "Protective" & option_type == "PUT" ~ p.rh1,
        
        
        strat == "Protective" & option_type == "CALL" ~ c.rh1,
        
        strat == "Strip" ~ c.rh1 + 2*p.rh1,
        strat == "Strap" ~ 2*c.rh1 + p.rh1,
        
        side == "LONG" & option_type == "CALL" & strat == "Butterfly" ~ c.rh1 - 2*c.rh2 + c.rh3,
        side == "SHORT" & option_type == "CALL" & strat == "Butterfly" ~ -(c.rh1 - 2*c.rh2 + c.rh3),
        
        side == "LONG" & option_type == "PUT" & strat == "Butterfly" ~ p.rh1 - 2*p.rh2 + p.rh3,
        side == "SHORT" & option_type == "PUT" & strat == "Butterfly" ~ -(p.rh1 - 2*p.rh2 + p.rh3),
        
        
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ c.rh1 - c.rh2 -c.rh3,
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ -(c.rh1 - c.rh2 -c.rh3),
        
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ p.rh1 + p.rh2 - p.rh3,
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ -(p.rh1 + p.rh2 - p.rh3),
        
        strat == "Jade Lizard" ~ -p.rh1 -c.rh2 + c.rh3,
        strat == "Reverse Jade Lizard" ~ p.rh1 -p.rh2 -c.rh3,
        
        
        side == "LONG" & option_type == "CALL" & strat == "Condor" ~ c.rh1 - c.rh2 - c.rh3 + c.rh4,
        side == "SHORT" & option_type == "CALL" & strat == "Condor" ~ -(c.rh1 - c.rh2 - c.rh3 + c.rh4),
        
        side == "LONG" & option_type == "PUT" & strat == "Condor" ~  p.rh1 - p.rh2 - p.rh3 + p.rh4,
        side == "SHORT" & option_type == "PUT" & strat == "Condor" ~ -(p.rh1 - p.rh2 - p.rh3 + p.rh4)
        
        
        
      ) %>% `*`(size) %>% round(., digits = 5),
      
      sts = case_when(
        
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ "Bullish",
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ "Bearish",
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ "Bearish",
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ "Bullish",
        strat == "Straddle" & side == "LONG" ~ "Bullish",
        strat == "Straddle" & side == "SHORT" ~ "Neutral",
        strat == "Strangle" & side == "LONG" ~ "Bullish",
        strat == "Strangle" & side == "SHORT" ~ "Neutral",
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ "Bullish",
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ "Bearish",
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ "Bullish",
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ "Bearish",
        strat == "Covered" & option_type == "PUT" ~ "Bearish",
        strat == "Covered" & option_type == "CALL" ~ "Bullish",
        strat == "Protective" & option_type == "PUT" ~ "Bullish",
        strat == "Protective" & option_type == "CALL" ~ "Bearish",
        strat == "Strip" ~ "Bearish",
        strat == "Strap" ~ "Bullish",
        strat == "Butterfly" ~ "Neutral",
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ "Bullish",
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ "Bearish",
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ "Bullish",
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ "Bearish",
        strat == "Jade Lizard" ~ "Bullish",
        strat == "Reverse Jade Lizard" ~ "Bearish",
        strat == "Condor" ~ "Neutral"
        
      ),
      risk = case_when(
        strat == "Single" & side == "LONG" & option_type == "CALL" ~ "Moderate Risk",
        strat == "Single" & side == "SHORT" & option_type == "CALL" ~ "High Risk",
        strat == "Single" & side == "LONG" & option_type == "PUT" ~ "Moderate Risk",
        strat == "Single" & side == "SHORT" & option_type == "PUT" ~ "High Risk",
        strat == "Straddle" & side == "LONG" ~ "High Risk",
        strat == "Straddle" & side == "SHORT" ~ "High Risk",
        strat == "Strangle" & side == "LONG" ~ "High Risk",
        strat == "Strangle" & side == "SHORT" ~ "High Risk",
        strat == "Spread" & side == "LONG" & option_type == "CALL" ~ "Moderate Risk",
        strat == "Spread" & side == "SHORT" & option_type == "CALL" ~ "Moderate Risk",
        strat == "Spread" & side == "SHORT" & option_type == "PUT" ~ "Moderate Risk",
        strat == "Spread" & side == "LONG" & option_type == "PUT" ~ "Moderate Risk",
        strat == "Covered" & option_type == "PUT" ~ "Moderate Risk",
        strat == "Covered" & option_type == "CALL" ~ "Low Risk",
        strat == "Protective" & option_type == "PUT" ~ "Moderate Risk",
        strat == "Protective" & option_type == "CALL" ~ "Low Risk",
        strat == "Strip" ~ "High Risk",
        strat == "Strap" ~ "High Risk",
        strat == "Butterfly" ~ "Low Risk",
        side == "LONG" & option_type == "CALL" & strat == "Ladder" ~ "High Risk",
        side == "SHORT" & option_type == "CALL" & strat == "Ladder" ~ "High Risk",
        side == "SHORT" & option_type == "PUT" & strat == "Ladder" ~ "High Risk",
        side == "LONG" & option_type == "PUT" & strat == "Ladder" ~ "High Risk",
        strat == "Jade Lizard" ~ "Moderate Risk",
        strat == "Reverse Jade Lizard" ~ "Moderate Risk",
        strat == "Condor" ~ "Low Risk"
      ),
      max_profit = case_when(
        # Single Option Strategies
        strat == "Single" & side == "LONG" ~ Inf,
        strat == "Single" & side == "SHORT" ~ max(y, na.rm = TRUE),
        
        # Straddle Strategy
        strat == "Straddle" & side == "LONG" ~ Inf,
        strat == "Straddle" & side == "SHORT" ~ max(y, na.rm = TRUE),
        
        # Strangle Strategy
        strat == "Strangle" & side == "LONG" ~ Inf,
        strat == "Strangle" & side == "SHORT" ~ max(y, na.rm = TRUE),
        
        strat == "Spread" ~ max(y, na.rm = TRUE),
        strat == "Covered"  ~ max(y, na.rm = TRUE),
        
        strat == "Protective" ~ Inf,
        
        strat %in% c("Strip","Strap") ~ Inf,
        
        strat == "Ladder" & side == "SHORT" ~ Inf,
        strat == "Ladder" & side == "LONG" ~ max(y, na.rm = TRUE),
        
        strat %in% c("Butterfly","Jade Lizard","Condor","Reverse Jade Lizard") ~ max(y, na.rm = TRUE),
        
      ),
      
      min_profit = case_when(
        # Single Option Strategies
        strat == "Single" & side == "LONG" ~ min(y, na.rm = TRUE),
        strat == "Single" & side == "SHORT" ~ -Inf,
        
        # Straddle Strategy
        strat == "Straddle" & side == "LONG" ~ min(y, na.rm = TRUE),
        strat == "Straddle" & side == "SHORT" ~ -Inf,
        
        # Strangle Strategy
        strat == "Strangle" & side == "LONG" ~ min(y, na.rm = TRUE),
        strat == "Strangle" & side == "SHORT" ~ -Inf,
        
        strat == "Spread"  ~ min(y, na.rm = TRUE),
        
        strat == "Covered" & option_type == "PUT" ~ -Inf,
        strat == "Covered" & option_type == "CALL" ~ -(s - price2),
        
        strat == "Protective" ~ min(y, na.rm = TRUE),
        
        strat %in% c("Strip","Strap") ~ min(y, na.rm = TRUE),
        
        strat == "Ladder" & side == "SHORT" ~ min(y, na.rm = TRUE),
        strat == "Ladder" & side == "LONG" ~ -Inf,
        
        strat %in% c("Jade Lizard","Reverse Jade Lizard") ~ -Inf,
        
        strat %in% c("Butterfly","Condor") ~ min(y, na.rm = TRUE),
        
      )
      
    )
  return(df)
}



server <- function(input, output, session) {
  
  
  observeEvent(input$fetch, {
    tryCatch({
      ticker <- input$ticker
      
      if (is.null(ticker) || ticker == "") {
        showNotification("Please enter a valid ticker symbol.", type = "error")
        return()
      }
      
      
      data <- py$fetch_data(ticker)
      vola <- py$get_52weeks_volatility(ticker)
      
      
      underlying_price <- data[[1]]
      rf <- data[[2]]
      dividend_yield <- data[[3]]
      expiry_dates <- data[[6]]
      open_price <- data[[7]]
      high <- data[[8]]
      low <- data[[9]]
      chg <- data[[10]]
      name <- data[[11]]
      sector <- data[[12]]
      vol <- vola
      
      
      if (any(sapply(c(underlying_price, rf, dividend_yield, expiry_dates, open_price, high, low, chg, name, sector), is.null)) || is.na(underlying_price)) {
        showNotification("Failed to retrieve all required data. Please check the ticker.", type = "error")
        return()
      }
      
      
      output$stock_name <- renderUI({
        tags$div(
          style = "font-size: 32px; font-weight: bold;",
          paste0(toupper(name), " (", toupper(ticker), ")")
        )
      })
      
      output$sector <- renderUI({
        tags$div(
          style = "font-size: 20px;",
          paste("SECTOR:", toupper(sector))
        )
      })
      
      
      output$open <- renderText({
        HTML(paste0("<span style = 'color black:'><b>Open: ",round(open_price, 2),"</b></span>"))
      })
      
      output$high <- renderText({
        HTML(paste0("<span style = 'color black:'><b>High: ",round(high, 2),"</b></span>"))
      })
      
      
      output$low <- renderText({
        HTML(paste0("<span style = 'color black:'><b>Low: ",round(low, 2),"</b></span>"))
      })
      
      output$close <- renderText({
        HTML(paste0("<span style = 'color black:'><b>Close: ",round(underlying_price, 2),"</b></span>"))
      })
      
      
      output$chg <- renderText({
        HTML(paste0("<span style = 'color black:'><b>Change: <b/></span>",
                    ifelse(chg < 0,
                           paste0("<span style='color: #fc0335;'><b>", round(chg, 2), "%</b></span>"),
                           paste0("<span style='color: #1cd4c8;'><b>+", round(chg, 2), "%</b></span>")
                    )
        ))
      })
      
      output$vol <- renderText({
        HTML(paste0("<span style = 'color black:'><b>Volatility(52w): ",round(vol, 2),"%</b></span>"))
      })
      
      
      
      updateNumericInput(session, "s", value = round(underlying_price, 2))
      updateNumericInput(session, "rf", value = ifelse(is.null(rf), 3.95, round(rf, 2)))
      updateNumericInput(session, "y", value = round(dividend_yield, 2))
      updateNumericInput(session, "size", value = 1)
      
      
      
      option_data_strike <- bind_rows(py_to_r(data[[4]]), py_to_r(data[[5]]))
      
      output$strike_ui <- renderUI({
        req(input$strat)  
        
        strikes <- sort(unique(option_data_strike$strike), decreasing = FALSE)
        
        #st2 : strangle, spread, strip, strap
        #st3 : butterfly, ladder, jade lizard
        #st4 : condor
        
        
        if(input$strat %in% c("Strangle","Spread")){
          tagList(
            selectInput("strike1",HTML(paste0("Strike ","<sub>1</sub>"," (k1; Lower k)")), choices = strikes),
            selectInput("strike2",HTML(paste0("Strike ","<sub>2</sub>"," (k2; Higher k)")), choices = strikes)
          )
          
        } else if(input$strat %in% c("Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard")) {
          tagList(
            selectInput("strike1",HTML(paste0("Strike ","<sub>1</sub>"," (k1; Even L k)")), choices = strikes),
            selectInput("strike2",HTML(paste0("Strike ","<sub>2</sub>"," (k2; Lower k)")), choices = strikes),
            selectInput("strike3",HTML(paste0("Strike ","<sub>3</sub>"," (k3; Higher k)")), choices = strikes)
          )
          
        } else if(input$strat %in% c("Condor")) {
          tagList(
            selectInput("strike1",HTML(paste0("Strike ","<sub>1</sub>"," (k1; Even L k)")), choices = strikes),
            selectInput("strike2",HTML(paste0("Strike ","<sub>2</sub>"," (k2; Lower k)")), choices = strikes),
            selectInput("strike3",HTML(paste0("Strike ","<sub>3</sub>"," (k3; Higher k)")), choices = strikes),
            selectInput("strike4",HTML(paste0("Strike ","<sub>4</sub>"," (k4; Even H k)")), choices = strikes)
          )
          
        } else {
          selectInput("strike", "Strike Price (k)", choices = strikes)
        }
      })
      
      updateSelectInput(session, "expiry", choices = expiry_dates)
      
      
      showNotification("Data fetched successfully!", type = "message", duration = 3)
      
    }, error = function(e) {
      showNotification(paste("Error: ", e$message), type = "error", duration = 3)
    })
  })
  
  observeEvent(input$expiry, {
    tryCatch({
      dte <- as.numeric(as.Date(input$expiry) - Sys.Date())
      
      if (is.na(dte) || dte <= 0) {
        showNotification("Please select a valid expiry date for options", type = "error", duration = 5)
        return()
      }
      
      updateNumericInput(session, "tau", value = dte)
    }, error = function(e) {
      showNotification(paste("Error: ", e$message), type = "error", duration = 5)
    })
  })
  
  observeEvent({
    input$s
    input$strike
    input$tau
    input$rf
    input$sigma
    input$y
    input$strat
    input$option_type
    
    if(input$strat %in% c("Strangle","Spread")) {
      list(input$strike1, input$strike2)
    } else if(input$strat %in% c("Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard")) {
      list(input$strike1, input$strike2, input$strike3)
    } else if(input$strat %in% c("Condor")) {
      list(input$strike1, input$strike2, input$strike3, input$strike4)
    } else {
      list(input$strike)
    }}
    
    ,{
      tryCatch({
        s <- input$s
        tau <- input$tau
        rf <- input$rf
        sigma <- input$sigma
        y <- input$y
        strategy <- input$strat
        option_type <- input$option_type
        
        if (strategy %in% c("Strangle","Spread")) {
          k1 <- as.numeric(input$strike1)
          k2 <- as.numeric(input$strike2)
          k3 <- NA
          k4 <- NA
          
        } else if (input$strat %in% c("Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard")) {
          k1 <- as.numeric(input$strike1)
          k2 <- as.numeric(input$strike2)
          k3 <- as.numeric(input$strike3)
          k4 <- NA
          
        } else if (input$strat %in% c("Condor")) {
          k1 <- as.numeric(input$strike1)
          k2 <- as.numeric(input$strike2)
          k3 <- as.numeric(input$strike3)
          k4 <- as.numeric(input$strike4)
          
        } else {
          k1 <- as.numeric(input$strike)
          k2 <- as.numeric(input$strike)
          k3 <- NA
          k4 <- NA
          
        }
        
        
        if (any(is.na(c(s, k1, tau, rf, sigma, y))) || tau <= 0 || sigma <= 0) {
          showNotification("All input values must be valid numbers (σ > 0, τ > 0).", type = "error", duration = 5)
          return()
        }
        
        cp1 <- bs_model(s, k1, rf, tau, sigma, y, "c")
        cp2 <- bs_model(s, k2, rf, tau, sigma, y, "c")
        cp3 <- bs_model(s, k3, rf, tau, sigma, y, "c")
        cp4 <- bs_model(s, k4, rf, tau, sigma, y, "c")
        
        pp1 <- bs_model(s, k1, rf, tau, sigma, y, "p")
        pp2 <- bs_model(s, k2, rf, tau, sigma, y, "p")
        pp3 <- bs_model(s, k3, rf, tau, sigma, y, "p")
        pp4 <- bs_model(s, k4, rf, tau, sigma, y, "p")
        
        
        output$price_ui <- renderUI({
          
          side <- input$side
          
          if (strategy == "Spread" & option_type == "CALL") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k1</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k2</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA))
            )
            
          } else if (strategy == "Spread" & option_type == "PUT") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k1</sub>")), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k2</sub>")), value = ifelse(!is.na(pp2), round(pp2, 3), NA))
            )
            
          } else if (strategy %in% c("Strip")){
            tagList(
              numericInput("st_price_1", HTML("<span style='color: #1cd4c8;'><b>+</b></span> 2x Put <sub>k</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML("<span style='color: #1cd4c8;'><b>+</b></span> Call <sub>k</sub>"), value = ifelse(!is.na(cp2), round(cp2, 3), NA))
            )
            
          } else if (strategy %in% c("Strap")){
            tagList(
              numericInput("st_price_1", HTML("<span style='color: #1cd4c8;'><b>+</b></span> Put <sub>k</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML("<span style='color: #1cd4c8;'><b>+</b></span> 2x Call <sub>k</sub>"), value = ifelse(!is.na(cp2), round(cp2, 3), NA))
            )
            
          } else if (strategy %in% c("Butterfly") & option_type == "CALL") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k1</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," 2x Call <sub>k2</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k3</sub>")), value = ifelse(!is.na(cp3), round(cp3, 3), NA))
            )
            
          } else if (strategy %in% c("Butterfly") & option_type == "PUT") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k1</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," 2x Put <sub>k2</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k3</sub>")), value = ifelse(!is.na(cp3), round(cp3, 3), NA))
            )
            
          } else if (strategy %in% c("Ladder") & option_type == "CALL") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k1</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k2</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k3</sub>")), value = ifelse(!is.na(cp3), round(cp3, 3), NA))
            )
            
          } else if (strategy %in% c("Ladder") & option_type == "PUT") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k1</sub>")), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k2</sub>")), value = ifelse(!is.na(pp2), round(pp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k3</sub>")), value = ifelse(!is.na(pp3), round(pp3, 3), NA))
            )
            
          } else if (strategy == "Jade Lizard") {
            tagList(
              numericInput("st_price_1", HTML("<span style='color: #d41c78;'><b>-</b></span> Put <sub>k1</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML("<span style='color: #d41c78;'><b>-</b></span> Call <sub>k2</sub>"), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML("<span style='color: #1cd4c8;'><b>+</b></span> Call <sub>k3</sub>"), value = ifelse(!is.na(cp3), round(cp3, 3), NA))
            )
            
          }  else if (strategy == "Reverse Jade Lizard") {
            tagList(
              numericInput("st_price_1", HTML("<span style='color: #1cd4c8;'><b>+</b></span> Put <sub>k1</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML("<span style='color: #d41c78;'><b>-</b></span> Put <sub>k2</sub>"), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML("<span style='color: #d41c78;'><b>-</b></span> Call <sub>k3</sub>"), value = ifelse(!is.na(cp3), round(cp3, 3), NA))
            )
            
          } else if (strategy == "Condor" & option_type == "CALL") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k1</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k2</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k3</sub>")), value = ifelse(!is.na(cp3), round(cp3, 3), NA)),
              numericInput("st_price_4", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k4</sub>")), value = ifelse(!is.na(cp4), round(cp4, 3), NA))
            )
            
          } else if (strategy == "Condor" & option_type == "PUT") {
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k1</sub>")), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k2</sub>")), value = ifelse(!is.na(pp2), round(pp2, 3), NA)),
              numericInput("st_price_3", HTML(paste0(ifelse(side == "SHORT",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k3</sub>")), value = ifelse(!is.na(pp3), round(pp3, 3), NA)),
              numericInput("st_price_4", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k4</sub>")), value = ifelse(!is.na(pp4), round(pp4, 3), NA))
            )
            
          } else if (strategy %in% c("Single")) {
            
            if(option_type == "CALL"){
              tagList(
                numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                              "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                              "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k</sub>")), value = ifelse(!is.na(cp1), round(cp1, 3), NA))
                
              )
              
            } else {
              tagList(
                numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                              "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                              "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k</sub>")), value = ifelse(!is.na(pp1), round(pp1, 3), NA))
                
              )
              
            }
            
          } else if (strategy %in% c("Covered")) {
            
            if(option_type == "CALL"){
              tagList(
                numericInput("st_price_1",HTML("S ","<span style='color: #d41c78;'><b>-</b></span> Call <sub>k</sub>"), value = ifelse(!is.na(cp1), round(cp1, 3), NA))
                
              )
              
            } else {
              tagList(
                numericInput("st_price_1",HTML(" - S ","<span style='color: #d41c78;'><b>-</b></span> Put <sub>k</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA))
                
              )
              
            }
            
          } else if (strategy %in% c("Protective")) {
            
            if(option_type == "CALL"){
              tagList(
                numericInput("st_price_1",HTML("- S ","<span style='color: #1cd4c8;'><b>+</b></span> Call <sub>k</sub>"), value = ifelse(!is.na(cp1), round(cp1, 3), NA))
                
              )
              
            } else {
              tagList(
                numericInput("st_price_1",HTML("S ","<span style='color: #1cd4c8;'><b>+</b></span> Put <sub>k</sub>"), value = ifelse(!is.na(pp1), round(pp1, 3), NA))
                
              )
              
            }
            
          }  else {
            # Default case for other strategies or option types
            tagList(
              numericInput("st_price_1", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Put <sub>k</sub>")), value = ifelse(!is.na(pp1), round(pp1, 3), NA)),
              numericInput("st_price_2", HTML(paste0(ifelse(side == "LONG",
                                                            "<span style='color: #1cd4c8;'><b>+</b></span>",
                                                            "<span style='color: #d41c78 ;'><b>-</b></span>")," Call <sub>k</sub>")), value = ifelse(!is.na(cp2), round(cp2, 3), NA))
            )
          }
          
        })
        
      }, error = function(e) {
        showNotification(paste("Error: ", e$message), type = "error", duration = 5)
      })
    })
  
  
  
  
  observeEvent(input$plot_button, {
    
    req(
      input$s,
      input$rf,
      input$tau,
      input$sigma,
      input$y,
      input$strat,
      if(input$strat %in% c("Strangle","Spread")) {
        list(input$strike1, input$strike2,
             input$st_price_1,input$st_price_2)
        
      } else if(input$strat %in% c("Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard")) {
        list(input$strike1, input$strike2, input$strike3,
             input$st_price_1,input$st_price_2,input$st_price_3)
        
      } else if(input$strat %in% c("Condor")) {
        list(input$strike1, input$strike2, input$strike3, input$strike4,
             input$st_price_1,input$st_price_2, input$st_price_3, input$st_price_4)
      } else {
        list(input$strike,
             input$st_price_1, input$st_price_2)
      },
      input$expiry,
      input$ticker,
      input$greeks,
      input$option_type,
      input$side,
      input$size
    )
    
    tryCatch({
      
      s <- input$s
      rf <- input$rf
      tau <- input$tau
      sigma <- input$sigma
      y <- input$y
      strategy <- input$strat
      greeks <- input$greeks
      option_type <- input$option_type
      side <- input$side
      size <- input$size
      ticker <- input$ticker
      expiry <- input$expiry
      
      
      if (strategy %in% c("Strangle","Spread")) {
        k1 <- as.numeric(input$strike1)
        k2 <- as.numeric(input$strike2)
        k3 <- NA
        k4 <- NA
        
        price1 <- input$st_price_1
        price2 <- input$st_price_2
        price3 <- NA
        price4 <- NA
        
      } else if (input$strat %in% c("Butterfly","Ladder","Jade Lizard","Reverse Jade Lizard")) {
        k1 <- as.numeric(input$strike1)
        k2 <- as.numeric(input$strike2)
        k3 <- as.numeric(input$strike3)
        k4 <- NA
        
        price1 <- input$st_price_1
        price2 <- input$st_price_2
        price3 <- input$st_price_3
        price4 <- NA
        
      } else if (input$strat %in% c("Condor")) {
        k1 <- as.numeric(input$strike1)
        k2 <- as.numeric(input$strike2)
        k3 <- as.numeric(input$strike3)
        k4 <- as.numeric(input$strike4)
        
        price1 <- input$st_price_1
        price2 <- input$st_price_2
        price3 <- input$st_price_3
        price4 <- input$st_price_4
        
      } else if (input$strat %in% c("Single","Covered","Protective")) {
        k1 <- as.numeric(input$strike)
        k2 <- NA
        k3 <- NA
        k4 <- NA
        
        price1 <- input$st_price_1
        price2 <- NA
        price3 <- NA
        price4 <- NA
        
      } else {
        k1 <- as.numeric(input$strike)
        k2 <- as.numeric(input$strike)
        k3 <- NA
        k4 <- NA
        
        price1 <- input$st_price_1
        price2 <- input$st_price_2
        price3 <- NA
        price4 <- NA
        
      }
      
      
      option_data_py <- py$option_chain(expiry, ticker)
      
      option_df <- bind_rows(
        py_to_r(option_data_py[[1]]) %>% mutate(type = "CALL"),
        py_to_r(option_data_py[[2]]) %>% mutate(type = "PUT")
      ) %>%
        select(contractSymbol, type, everything())
      
      
      grk_df <- greek_df(option_df, s, rf, y, sigma, k1, k2, k3, k4, price1, price2, price3, price4, size, tau, option_type, side, strategy)
      
      grk_df <- grk_df %>%
        mutate(
          loss = case_when(
            y <= 0 ~ y,
            TRUE ~ NA
          ),
          profit = case_when(
            y >= 0 ~ y,
            TRUE ~ NA
          )
        )
      
      value <- grk_df %>%
        filter(x==round(s,digits = 1))
      
      pl <- value %>%
        pull(y)
      
      Delta <- value %>%
        pull(Delta)
      
      Gamma <- value %>%
        pull(Gamma)
      
      Vega <- value %>%
        pull(Vega)
      
      Theta <- value %>%
        pull(Theta)
      
      Rho <- value %>%
        pull(Rho)
      
      output$greeks_table <- renderUI({
        
        Head <- ifelse(strategy == "Single",
                       paste(side,option_type),
                       ifelse(strategy %in% c("Covered","Protective"),
                              paste(toupper(strategy),option_type),
                              
                              ifelse(strategy %in% c("Straddle","Strangle"),
                                     paste(side,toupper(strategy)),
                                     
                                     ifelse(strategy %in% c("Strip","Strap","Jade Lizard","Reverse Jade Lizard"),
                                            paste(toupper(strategy)),
                                            paste(side,option_type,toupper(strategy))
                                     ))))
        
        
        status <- grk_df %>%
          pull(sts) %>%
          unique() %>%
          map(~ case_when(
            . == "Bullish" ~ HTML(paste0("<code style='color:#f05f3e;'>", ., "</code>")),
            . == "Bearish" ~ HTML(paste0("<code style='color:#34b4eb;'>", ., "</code>")),
            TRUE ~ HTML(paste0("<code style='color:#59d9b5;'>", ., "</code>"))
          ))
        
        
        risk <- grk_df %>%
          pull(risk) %>%
          unique() %>%
          map(~ case_when(
            . == "High Risk" ~ HTML(paste0("<code style='color:#f03ec6;'>", ., "</code>")),
            . == "Low Risk" ~ HTML(paste0("<code style='color:#3ebaf0;'>", ., "</code>")),
            TRUE ~ HTML(paste0("<code style='color:#37ad8c;'>", ., "</code>"))
          ))
        
        HTML(
          paste(
            "<br>",
            "<span style ='font-size:180%; text-align:left;'><b> STRATEGY:",Head,"</b></span>",
            "<br>",
            status,"&nbsp;",risk,
            "<br>",
            "<p style='text-align:center; font-weight:bold;'>","<br>",
            
            "P/L (<span style='color:#606375;'>@S</span>)",
            "<span style='color:#606375;'>", format(pl*100, nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "Delta(<span style='color:##09a0ad;'>Δ</span>)",
            "<span style='color:##09a0ad;'>", format(round(Delta,digits=3), nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "Gamma(<span style='color:#9d03fc;'>Γ</span>)",
            "<span style='color:#9d03fc;'>", format(round(Gamma,digits=3), nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "Vega(<span style='color:#fc0394;'>ν</span>)",
            "<span style='color:#fc0394;'>", format(round(Vega,digits=3), nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "Theta(<span style='color:#fc4503;'>θ</span>)",
            "<span style='color:#fc4503;'>", format(round(Theta,digits=3), nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "Rho(<span style='color:#0324fc;'>ρ</span>)",
            "<span style='color:#0324fc;'>", format(round(Rho,digits=3), nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "IV(<span style='color:#5b376b;'>σ</span>)",
            "<span style='color:#5b376b;'>", format(NA, nsmall = 3), "</span>",
            "&nbsp;&nbsp;&nbsp;",
            
            "</p>",
            "<hr style='border-top: 1px solid rgba(0,0,0,0.1);'/>"
          )
        )
        
      })
      
      
      output$strategy_plot <- renderPlotly({
        
        p <- grk_df %>%
          plot_ly() %>%
          add_trace(
            x = ~x,
            y = ~profit,
            type = 'scatter',
            mode = 'lines',
            fill = 'tozeroy',
            name = 'Profit',
            line = list(color = 'skyblue'),
            fillcolor = 'rgba(135, 206, 235, 0.25)',
            yaxis = "y"
          ) %>%
          add_trace(
            x = ~x,
            y = ~loss,
            type = 'scatter',
            mode = 'lines',
            fill = 'tozeroy',
            name = 'Loss',
            line = list(color = 'red'),
            fillcolor = 'rgba(255, 0, 0, 0.25)',
            yaxis = "y"
          ) %>%
          layout(
            title = HTML(paste0("<span style><b>[",toupper(ticker), "] ","OPTION P/L </b></span>")),
            xaxis = list(
              title = "PRICE",
              showgrid = TRUE,
              gridcolor = 'rgba(200, 200, 200, 0.3)',
              showspikes = TRUE,  
              spikethickness = 0.6, 
              spikecolor = 'rgba(120, 120, 120, 0.7)',  
              spikedash = 'line'
            ),
            yaxis = list(
              title = "P/L",
              zeroline = FALSE,
              showgrid = TRUE,
              gridcolor = 'rgba(200, 200, 200, 0.3)'
            ),
            yaxis2 = list(
              title = "Greeks",
              overlaying = "y",
              side = "right",
              zeroline = FALSE,
              fixedrange = TRUE,
              showgrid = FALSE
            ),
            hovermode = "x unified",  
            hoverlabel = list(
              font = list(size = 10),
              bordercolor = 'rgba(0, 0, 0, 0)',
              bgcolor = 'rgba(255, 255, 255, 0.5)'
            ),
            shapes = list(
              list(
                type = 'line',
                x0 = s, x1 = s,
                y0 = min(grk_df$y), y1 = max(grk_df$y),
                line = list(dash = 'dot', color = 'grey', width = 0.5),
                xref = 'x', yref = 'y'
              ),
              list(
                type = 'line',
                x0 = s, x1 = s,
                y0 = min(grk_df$y), y1 = max(grk_df$y),
                line = list(dash = 'dot', color = 'grey', width = 0.5),
                xref = 'x', yref = 'y'
              )),
            annotations = list(
              x = s, y = max(grk_df$y),
              text = paste0("Price", ":$", s),
              xanchor = 'left', yanchor = 'top',
              showarrow = FALSE,
              font = list(color = 'grey')
            ),
            showlegend = TRUE,
            dragmode = "zoom",
            paper_bgcolor = 'white',
            plot_bgcolor = 'white'
          )
        
        
        if (greeks == "Delta") {
          p <- p %>%
            add_trace(
              data = grk_df,
              x = ~x,
              y = ~Delta,
              type = 'scatter',
              mode = 'lines',
              name = 'Delta',
              line = list(dash = 'dot', color = 'orange'),
              yaxis = "y2"
            )
        } else if (greeks == "Gamma") {
          p <- p %>%
            add_trace(
              data = grk_df,
              x = ~x,
              y = ~Gamma,
              type = 'scatter',
              mode = 'lines',
              name = 'Gamma',
              line = list(dash = 'dot', color = 'orange'),
              yaxis = "y2"
            )
        } else if (greeks == "Vega") {
          p <- p %>%
            add_trace(
              data = grk_df,
              x = ~x,
              y = ~Vega,
              type = 'scatter',
              mode = 'lines',
              name = 'Vega',
              line = list(dash = 'dot', color = 'orange'),
              yaxis = "y2"
            )
        } else if (greeks == "Theta") {
          p <- p %>%
            add_trace(
              data = grk_df,
              x = ~x,
              y = ~Theta,
              type = 'scatter',
              mode = 'lines',
              name = 'Theta',
              line = list(dash = 'dot', color = 'orange'),
              yaxis = "y2"
            )
        } else if (greeks == "Rho") {
          p <- p %>%
            add_trace(
              data = grk_df,
              x = ~x,
              y = ~Rho,
              type = 'scatter',
              mode = 'lines',
              name = 'Rho',
              line = list(dash = 'dot', color = 'orange'),
              yaxis = "y2"
            )
        }
        
        p
        
      })
      
      max_profit = ifelse(is.infinite(grk_df$max_profit),
                          "+\u221E",
                          grk_df$max_profit %>% `*`(100) %>% round(.,digits=2)) %>%
        unique()
      
      min_profit = ifelse(is.infinite(grk_df$min_profit),
                          "-\u221E",
                          grk_df$min_profit %>% `*`(100) %>% round(.,digits=2)) %>%
        unique()
      
      
      
      output$bep_value <- renderUI({
        HTML(paste("<p style='text-align:center; font-weight:bold;'>","<br>",
                   
                   "<span style='color:black;'>Profit <sub>Max</sub>:</span> ",
                   "<span style='color:rgb(0, 204, 153);'>", max_profit, " pt</span>",
                   "&nbsp;&nbsp;&nbsp;&nbsp;",
                   
                   "<span style='color:black;'>Loss <sub>Max</sub>:</span> ",
                   "<span style='color:rgb(255, 102, 102);'>", min_profit, " pt</span>",
                   "&nbsp;&nbsp;&nbsp;&nbsp;",
                   
                   "BEP<sub>1</sub>: ", unique(grk_df$bep1)  %>% round(.,digits=2), " pt",
                   "&nbsp;&nbsp;&nbsp;&nbsp;",
                   
                   "BEP<sub>2</sub>: ", ifelse(is.na(grk_df$bep2),"—",grk_df$bep2 %>% round(.,digits=2)) %>% unique(), " pt",
                   "</p>"))
      })
      
    }, error = function(e) {
      showNotification(paste("Error: ", e$message), type = "error", duration = 5)
    })
  })
  
  
  output$strategy_plot <- renderPlotly({
    NULL
  })
  
}


shinyApp(ui = ui, server = server)

