update hino set nome = 'MUITO EM BREVE O REI VEM', nome_pt = 'MUITO EM BREVE O REI VEM' where numero = 1 and coletanea_id = 3;
update hino set nome = 'PREPARAI-VOS, Ó IGREJA AMADA', nome_pt = 'PREPARAI-VOS, Ó IGREJA AMADA' where numero = 2 and coletanea_id = 3;
update hino set nome = 'JESUS VOLTARÁ', nome_pt = 'JESUS VOLTARÁ' where numero = 3 and coletanea_id = 3;
update hino set nome = 'AS PARÁBOLAS DO REINO', numero = 4, nome_pt = 'AS PARÁBOLAS DO REINO', numero = 4 where coletanea_id = 3;
update hino set nome = 'EU CLAMO, EU ORO', nome_pt = 'EU CLAMO, EU ORO' where numero = 5 and coletanea_id = 3;
update hino set nome = 'A VITÓRIA DA IGREJA', nome_pt = 'A VITÓRIA DA IGREJA' where numero = 6 and coletanea_id = 3;
update hino set nome = 'ESTÁ CHEGANDO A HORA', nome_pt = 'ESTÁ CHEGANDO A HORA' where numero = 7 and coletanea_id = 3;
update hino set nome_pt = nome where nome_pt is null;
update hino set idioma = 'PT-BR' where idioma is null;