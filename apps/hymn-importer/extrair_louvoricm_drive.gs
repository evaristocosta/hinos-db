function listarArquivosNaPasta() {
  // ID da pasta raiz que você quer listar (cole aqui o que aparece na URL do Drive depois de /folders/)
  var rootFolderId = '1G7fySQbj4merD9h0vla9CyZZFcMdK5YX';
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  //sheet.clear();
  sheet.appendRow(['Caminho', 'Nome', 'Tipo', 'URL']);
  listarArquivosRecursivo(DriveApp.getFolderById(rootFolderId), '', sheet);
}

/*function listarArquivosRecursivo(pasta, caminho, sheet) {
  var subCaminho = caminho + '/' + pasta.getName();
  // Arquivos na pasta atual
  var arquivos = pasta.getFiles();
  while (arquivos.hasNext()) {
    var arquivo = arquivos.next();
    sheet.appendRow([
      subCaminho,
      arquivo.getName(),
      'Arquivo',
      arquivo.getUrl()
    ]);
  }
  // Subpastas na pasta atual
  var subPastas = pasta.getFolders();
  while (subPastas.hasNext()) {
    var subPasta = subPastas.next();
    sheet.appendRow([
      subCaminho,
      subPasta.getName(),
      'Pasta',
      subPasta.getUrl()
    ]);
    listarArquivosRecursivo(subPasta, subCaminho, sheet);
  }
}*/

function listarArquivosRecursivo(pasta, caminho, sheet) {
  var subCaminho = caminho + '/' + pasta.getName();

  // Arquivos (lista e ordena normalmente)
  var arquivosArr = [];
  var arquivos = pasta.getFiles();
  while (arquivos.hasNext()) arquivosArr.push(arquivos.next());
  arquivosArr.sort(function (a, b) {
    return a.getName().localeCompare(b.getName());
  });
  for (var i = 0; i < arquivosArr.length; i++) {
    var arquivo = arquivosArr[i];
    sheet.appendRow([
      subCaminho,
      arquivo.getName(),
      'Arquivo',
      arquivo.getUrl(),
    ]);
  }

  // Subpastas (lista, ordena e filtra)
  var subPastasArr = [];
  var subPastas = pasta.getFolders();
  while (subPastas.hasNext()) subPastasArr.push(subPastas.next());
  subPastasArr.sort(function (a, b) {
    return a.getName().localeCompare(b.getName());
  });

  for (var j = 0; j < subPastasArr.length; j++) {
    var subPasta = subPastasArr[j];
    // ***Filtro: Pular pastas cujo nome começa com "A" ou "a"***
    if (/^a\d/i.test(subPasta.getName())) continue;
    sheet.appendRow([
      subCaminho,
      subPasta.getName(),
      'Pasta',
      subPasta.getUrl(),
    ]);
    listarArquivosRecursivo(subPasta, subCaminho, sheet);
  }
}
