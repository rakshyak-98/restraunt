function botScroll(){
    setTimeout(
    window.scrollTo({
    top: document.body.scrollHeight,
    behavior: 'smooth'
}),1000);
}
function unique(){
    alert("email already present. Please login");
}
function absent(){
    alert("Username/password incorrect")
}