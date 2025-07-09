# Project Proposal: Analyzing the U.S. Aviation Network
### Course: CUNY DATA620 Web Analytics
### Team Members: Ari & Lucas

## 1. Overview and Motivation

The U.S. aviation system is a complex network where local events, like bad weather, can cause widespread flight disruptions. Our project aims to model this network by combining flight route data with historical weather information. By building a knowledge graph, we will analyze how an airport's importance and local weather are related to flight disruptions, applying network analysis and machine learning to a real-world problem.

## 2. Guiding Question and Hypothesis

Guiding Question: Can we predict flight disruptions using an airport's importance in the flight network and its local weather?

Hypothesis: We believe that flights at major, well-connected airports are more likely to be cancelled during bad weather than flights at smaller airports with similar weather conditions.

## 3. Data Sources

Airline Route Data: From OpenFlights.org, this gives us airport, airline, and route information. We'll focus on the U.S. network.

Historical Weather Data: From NOAA, providing daily weather summaries (e.g., temperature, precipitation) for airport locations.

Flight Performance Data (Optional): From the Bureau of Transportation Statistics (BTS). We may use this for delay/cancellation data if time permits, but it is a stretch goal.

## 4. Plan of Action

Prepare Data: Download and clean the route and weather datasets.

Build Knowledge Graph: Set up a free Neo4j AuraDB instance and load the data. We'll create nodes for :Airport and :WeatherEvent and model the relationships between them.

Analyze Network: Use Cypher queries to find the most important airports (e.g., those with the most routes).

Train ML Model: Use the graph data (airport importance, weather) to train a model that predicts flight cancellations.

## 5. Machine Learning Goal

Our primary goal is to predict flight cancellations. Using the network and weather data, we will train a model to predict if a flight on a given route is likely to be cancelled. If we manage to integrate the optional BTS data, we may try to predict significant delays instead.

## 6. Potential Challenges

Data Merging: Combining the different datasets accurately will require careful work.

Data Scale: The datasets can be large, so we may need to limit our analysis to a specific timeframe or a subset of airports.

Feature Creation: Choosing the right features from our graph for the machine learning model will be a key task.