const main = document.querySelector('div');


let htmlString = "";

for ( let i = 10; i <= 10; i ++) {
htmlString += `<div>${i}</div>`;
}
main.innerHTML = htmlString; 
console.log(htmlString);