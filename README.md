Per ricostruire il db partendo dai frammenti usa il seguente comando dalla main directory del progetto:

```
cat data/db/italian_stocks.db.part* > data/db/italian_stocks.db
```

Per lanciare il DB plotter, lancia il seguente comando dalla main directory del progetto:

```
streamlit run src/data/db_plotter.py
```