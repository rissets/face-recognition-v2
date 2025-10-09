<template>
  <div>
    <NavigationBar />
    <main class="container">
      <RouterView />
    </main>
  </div>
</template>

<script setup>
import NavigationBar from './components/NavigationBar.vue'
</script>
