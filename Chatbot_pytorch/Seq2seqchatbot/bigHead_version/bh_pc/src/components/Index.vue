<template>
  <div class="box">
    <Header></Header>
    <div class="info">
      <h1 style="font-size:60px;">和AI大头聊天吧！</h1>
      <div class='detail'>游客每日可聊天三次</div>
      <div class='detail'>用户每日聊天次数无上限</div>
      <div class='detail'>并能拥有聊天记录等个性化功能</div>
    </div>
    <div class="go">
      <router-link to="/chat">
        <el-button round> > > GO!</el-button>
      </router-link>
    </div>
    <span class="example">
      <img src="../../static/image/sample.png" alt="NO EXISTS">
    </span>
  </div>
</template>

<script>
import Header from "./common/Header"
import Chat from "./Chat";

export default {
  name: "Index",
  data() {
    return {}
  },
  created() {
  },
  mounted(){
  },
  methods: {

  },
  components: {
    Header,
    Chat,
  }
}
</script>

<style scoped>
.box {
  background-color: #b5c6c6;
  position: fixed;
  width: 100%;
  height: 100%;
}

.info {
  color: #465657;
  position: absolute;
  left: 200px;
  top: 200px;
}

.detail {
  font-style: italic;
  padding-top: 15px;
  padding-left: 20px;
  text-shadow: 3px 3px 10px #9cacb6;
}

.go {
  position: absolute;
  height: 230px;
  bottom: 0;
  left: 570px;
}

.el-button {
  background-color: #465657;
  border: none;

  /*按钮内字颜色*/
  color: #b5c6c6;
  font-size: large;
  font-weight: bold;

  cursor: pointer;
}

.el-button :hover {
  color: #d1e6e6;
}

.example {
  position: relative;
  top: 100px;
}

img {
  width: 25%;
  position: relative;
  bottom: 70px;
  left: 780px;
}
</style>
