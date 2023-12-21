#############code 1: without data cleaning########
def market_basket_analysis(transaction_data, min_support=0.02, min_confidence=0.5):
    # Step 1: Organize shopping basket data
    basket_data = pd.DataFrame(transaction_data)

    # Step 2: Association rule mining
    frequent_itemsets = apriori(basket_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Step 3: Set thresholds
    support_threshold = min_support
    confidence_threshold = min_confidence
    # Step 4: Generate recommendation rules
    filtered_rules = rules[(rules['support'] >= support_threshold) & (rules['confidence'] >= confidence_threshold)]

    # Step 5: Real-time recommendation generation
    def generate_recommendations(user_basket, filtered_rules):
        recommendations = set()
        for _, rule in filtered_rules.iterrows():
            if set(rule['antecedents']).issubset(user_basket) and rule['consequents']:
                recommendations.update(rule['consequents'])
        return list(recommendations)

    return generate_recommendations


# Example Usage
transaction_data = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter', 'Cheese'],
    ['Milk', 'Butter', 'Eggs'],
    ['Bread', 'Cheese'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

generate_recommendations = market_basket_analysis(transaction_data, min_support=0.2, min_confidence=0.5)

user_basket = ['Milk', 'Bread']
recommendations = generate_recommendations(user_basket, filtered_rules)

print("User Basket:", user_basket)
print("Recommendations:", recommendations)


#############code 2: basket analysis engine########

'''
parameters_key = \
{conn_strv : '******',
transaction_nm : 'trans_market_basket',
item_key : 'product_type',
transaction_key : 'invoiceno',
qty : 'quantity',
trans_date : 'trans_date',
dropna_columns : 'invoiceno',
item_num_per_transaction_lbound : 2,
item_num_per_transaction_ubound	: 500,
lift_bound : 2,
confidence_bound : 0.5,
filter_cat : 'product_type',
filter_cat_levels : 'TOP_23',
transaction_filter : ["quantity > 0", "stockcode.str.contains('C') == False"],
timeformat : 'YYYY-MM-DD',
date_range : ['2011-01-15',  '2011-11-02']}


'''
def basket_analysis(parameters_key):
            
    transaction_nm, item_key, transaction_key,\
    qty, trans_date, dropna_columns, item_num_per_transaction_lbound,\
    item_num_per_transaction_ubound, lift_bound,\
    confidence_bound, filter_cat, filter_cat_levels,\
    transaction_filter, timeformat, date_range = parameters_key.values()
    
    # read raw data into df from db 
    engine = create_engine(conn_strv)

    date_st, date_ed = eval(date_range)
    
    tm_res0 = "((to_date('" + date_ed + "', '" + timeformat + "') - to_date("\
                + trans_date + ", '" + timeformat + "')) > 0) "    
                    
    tm_res1 = "((to_date('" + date_st + "', '" + timeformat + "') - to_date("\
               + trans_date + ", '" + timeformat + "')) < 0) "                    
    
    sql = "select " + transaction_nm + "." + filter_cat + ", count(*) as cnt from " +\
           transaction_nm + " where " + tm_res0  + " and " + tm_res1 +\
           " group by " + transaction_nm + "." + filter_cat + " order by cnt desc "
           
     
    data_tmp = pd.read_sql_query(sql, engine)
    data_tmp.shape

    try:
        cnt_threshold = int(filter_cat_levels[4:])
    except:  
        cnt_threshold = 1000
                    
    try:
        cats = str(tuple(eval(filter_cat_levels)))
    except:  
        cats = None
        
    cnt_df = data_tmp.sort_values(['cnt'], ascending = False)
    cnt_df.columns = [filter_cat, 'N']
    cnt_df.shape
    if cats == None:
        cnt_df = cnt_df.head(cnt_threshold)
        lst = str(tuple(list(cnt_df[filter_cat]))) 
    else:
        lst = cats[:]
    
    sql = "select * from " + transaction_nm + " where " +\
           tm_res0  + " and " + tm_res1 +\
           " and " + filter_cat + " in " +  lst + \
           " order by  " + transaction_key + ", " + item_key
               
    transaction_table = pd.read_sql_query(sql, engine)
    
    list(transaction_table.columns)
    transaction_table.shape       
    
    transaction_table.product_type.value_counts()
                     
    # dropna
    transaction_table.dropna(axis=0, subset=dropna_columns, inplace=True)
    
    # data type to str, clean data
    transaction_table[item_key] = transaction_table[item_key].astype(str)
    transaction_table[item_key] = transaction_table[item_key].str.strip()
    transaction_table[item_key] = transaction_table[item_key].str.replace(" ",'')
    transaction_table[item_key] = transaction_table[item_key].str.replace(",",'')
    transaction_table[item_key] = transaction_table[item_key].str.replace("'",'')
    transaction_table[item_key] = transaction_table[item_key].str.replace('"','')
    transaction_table[item_key] = transaction_table[item_key].str.strip()
    transaction_table[transaction_key] = transaction_table[transaction_key].astype('str')
    
    # conduct transaction data filters
    
    LL = 0
    conds = 1
    try:
        conds = eval(transaction_filter)
        if type(conds) == list:
            LL = len(conds)
    except:    
        conds = None
    
    if LL > 0 and (conds != None):
        con_str = ""
        for it in conds:
            it = "(transaction_table." + it + ")"
            if len(con_str)<1:
                con_str = it
            else:
                con_str = con_str + " & " + it
            
        transaction_table_new = transaction_table[eval(con_str)]        
             
    #transaction_table.shape
    #transaction_table_new.shape

    # basket analysis
    basket = (transaction_table_new.groupby([transaction_key, item_key])[qty]\
          .sum().unstack().reset_index().fillna(0).set_index(transaction_key))
    
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
    
    # further filter on items # in one transaction    
    basket_sets = basket.applymap(encode_units)
    basket_sets = basket_sets[(basket_sets > 0).sum(axis=1) >= item_num_per_transaction_lbound]
    basket_sets = basket_sets[(basket_sets > 0).sum(axis=1) <= item_num_per_transaction_ubound]
    
    frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    #rules.head()
    rules = rules[(rules['lift'] >= lift_bound) & (rules['confidence'] >= confidence_bound)]
    
    return [rules, frequent_itemsets]






