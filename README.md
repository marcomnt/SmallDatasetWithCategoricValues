# SmallDatasetWithCategoricValues
Arquivos Novos:
     forestfires.csv      -- database usado no paper, ainda n foi tratado conforme o paper
     servo.csv             -- database usado no paper, é tratado pelo meu programa na execução
     test.csv               -- database criado por mim para testes, é bem simples e pequeno
                                                   X: label1{ A, B, C }, label2 { A, B, C, D }
                                                   Y: [ 0.0 : 1.0 ]
     Arguments,py    -- classe para representar os argumentos da função Fuzzy
                                                  L: atributo representando o argumento L da função Fuzzy
                                                  C: atributo representando o argumento C da função Fuzzy
                                                  U: atributo representando o argumento U da função Fuzzy
                                                  calculate(x): metodo para calcular a função dado um x, retorna um float
     paper.pdf          -- o paper upado no git
Atualizando Arquivos:
     marco.py           -- adicionado o makeCOBmat( retornando um interável ), e makeFyzzyProbabilisticFunction (retornando,
                                  por enquanto, uma lista de Arguments) o proimo passo é fazer a função retornar uma dupla (uma                   
                                  para a lista de Arguments dos valores de X e outra para o Argument de Y)
