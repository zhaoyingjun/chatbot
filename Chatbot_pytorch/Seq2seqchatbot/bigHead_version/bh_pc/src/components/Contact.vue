<template>
  <div class="box">
    <Header></Header>
    <div class="info-box">
      <div class="contact">
        <h1 style="font-size: 20px; margin: 15px 10px;"> 联系邮箱</h1>
        <div> n4atz@hotmail.com </div>
        <h1 style="font-size: 20px; margin: 15px 10px;"> QQ </h1>
        <div> 414896749 </div>
        <div class="text">欢迎联系我们(/≧▽≦)/</div>
      </div>
    </div>
  </div>
</template>

<script>
import Header from "./common/Header";

export default {
  name: "Contact",
  data() {
    return {}
  },
  methods: {},
  components: {
    Header,
  }
}
</script>

<style scoped>
.box{
  position: fixed;
  background-color: #b4c0d0;
  height:100%;
  width:100%;
}

.info-box{
  position: absolute;
  top:30%;
  left:36%;
  height: 300px;
  width: 400px;

  background-color: #d3dce6;
  color: #47555e;
  box-shadow: 0 2px 20px 0 rgba(0, 0, 0, 0.1);

  border-radius: 10px;
  border: 1px solid;
  border-color: #D3DCE6;

  text-align: center;
}

.contact{
  position: relative;
  top:10%;
}

.text{
  position: relative;
  top: 40px;
  left: 10px;
  font-size:25px;
  letter-spacing:2px;
}
</style>
