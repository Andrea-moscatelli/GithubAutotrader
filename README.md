Per ricostruire il db partendo dai frammenti usa il seguente comando dalla main directory del progetto:

```
cat data/db/italian_stocks.db.part* > data/db/italian_stocks.db
```

Per lanciare il DB plotter, lancia il seguente comando dalla main directory del progetto:

```
streamlit run src/data/db_plotter.py
```


prossimi passi:
- FATTO - definire ticks_retriever_interactive_broker.py che raccoglie i dati storici delle azioni italiane e li salva nel db
- FATTO - far in modo che tickers.csv abbia la struttura corretta (per ticks_retriever_interactive_broker.py)
- integra cio che fa from_ticker_csv_to_table_ib in ticks_retriever_interactive_broker.py
- testa lo script e con i primi 50 tickers e che vengano registrati i contratti invalidi
- definire ticks_selector.py che seleziona le migliori X azioni che verranno usate per il live trading
- definire live_trader.py che esegue il trading in tempo reale sulle azioni selezionate
- dockerizzare l'intero progetto
- estendere il progetto ad altri mercati

