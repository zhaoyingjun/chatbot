<template>
  <div class="box">
    <Header></Header>
    <div class="text">见Github</div>
  </div>
</template>

<script>
import Header from "./common/Header"

export default {
  name: "Doc",
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

  text-align: center;
}

.text{
  position: relative;
  top:35%;
  margin-bottom: 20px;
  color: #47555e;
  font-size: 30px;
  letter-spacing:5px;
}
</style>
